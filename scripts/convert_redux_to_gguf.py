#!/usr/bin/env python3
"""Convert moondream/parakeet-redux (thrush-ternary-v2 safetensors) to GGUF.

The redux checkpoint ships every encoder linear/conv weight pre-packed:
  - `X.qweight`  [out, ceil(in/5)]           uint8  — base-3 digits, 5 per byte
  - `X.scales`   [out, in/128]               fp16   — one scale per 128-group
with  w[r, c] = scales[r, c//128] * (code - 1), code in {0,1,2}.

This emits those weights as GGML_TYPE_TRQ1_0 (block 256 = two 128-groups,
repacked from 5-digits/byte to the strided-32 4-digits/byte layout: 64
packed-digit bytes + 2 fp16 scales = 68 bytes/block, 2.125 bpw), so the C++ engine
dequantizes inside ggml_mul_mat with the new vec_dot. Everything else (decoder,
norms, convs [as_conv1d], VAD head, featurizer) is emitted F32 as the loader
expects. Tensor names are kept verbatim from the checkpoint minus the
`.qweight`/`.scales` suffix -> `.weight`.
"""
import argparse
import json
import pathlib
import struct

import numpy as np

try:
    import gguf
    from gguf.constants import GGMLQuantizationType
except ImportError as e:
    print(f"converter: missing dependency 'gguf': {e}", file=sys.stderr)
    sys.exit(2)

# ---- register the new ggml type with the gguf writer -----------------------
GGML_TYPE_TRQ1_0 = 42
GGMLQuantizationType.TRQ1_0 = GGML_TYPE_TRQ1_0
# (block_elements, block_bytes): 256-element block = two 128-groups; each group
# = 32 bytes of 4 base-3 digits (strided-32: byte m -> elements m, m+32, ...)
# + 2 fp16 scales = 68 bytes
import gguf.constants as _gc
_gc.GGML_QUANT_SIZES[GGML_TYPE_TRQ1_0] = (256, 68)


# ---- HF -> NeMo tensor-name remap (the engine reads NeMo state_dict names) --
# redux ships transformers-style names; parakeet.cpp expects the stock NeMo
# naming of parakeet-tdt-0.6b-v3.
_SELF_ATTN = {
    "q_proj": "linear_q", "k_proj": "linear_k", "v_proj": "linear_v",
    "o_proj": "linear_out", "relative_k_proj": "linear_pos",
    "bias_u": "pos_bias_u", "bias_v": "pos_bias_v",
}


def remap(name):
    """Map a redux tensor name to the NeMo name the engine reads."""
    # self-attn projections
    for hf, nemo in _SELF_ATTN.items():
        if f".self_attn.{hf}" in name:
            return name.replace(f".self_attn.{hf}", f".self_attn.{nemo}", 1)
    # conv batch_norm
    if ".conv.norm." in name:
        return name.replace(".conv.norm.", ".conv.batch_norm.", 1)
    # subsampling -> pre_encode conv/out
    if "encoder.subsampling.layers.0." in name:
        return name.replace("encoder.subsampling.layers.0.", "encoder.pre_encode.conv.0.", 1)
    if "encoder.subsampling.layers.2." in name:
        return name.replace("encoder.subsampling.layers.2.", "encoder.pre_encode.conv.2.", 1)
    if "encoder.subsampling.layers.3." in name:
        return name.replace("encoder.subsampling.layers.3.", "encoder.pre_encode.conv.3.", 1)
    if "encoder.subsampling.layers.5." in name:
        return name.replace("encoder.subsampling.layers.5.", "encoder.pre_encode.conv.5.", 1)
    if "encoder.subsampling.layers.6." in name:
        return name.replace("encoder.subsampling.layers.6.", "encoder.pre_encode.conv.6.", 1)
    if "encoder.subsampling.linear." in name:
        return name.replace("encoder.subsampling.linear.", "encoder.pre_encode.out.", 1)
    # decoder lstm / embed
    if "decoder.lstm." in name:
        return name.replace("decoder.lstm.", "decoder.prediction.dec_rnn.lstm.", 1)
    if "decoder.embedding." in name:
        return name.replace("decoder.embedding.", "decoder.prediction.embed.", 1)
    # joint / projections
    if "decoder.decoder_projector." in name:
        return name.replace("decoder.decoder_projector.", "joint.pred.", 1)
    if "encoder_projector." in name:
        return name.replace("encoder_projector.", "joint.enc.", 1)
    if "joint.head." in name:
        return name.replace("joint.head.", "joint.joint_net.2.", 1)
    return name


def read_safetensors(path):
    """Read a safetensors single-file checkpoint; returns {name: data_bytes, dtype, shape} + offsets lazily."""
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
        data = fh.read()  # rest is raw tensor bytes
    return header, data


def load_tensor(fh, meta, data):
    start, end = meta["data_offsets"]
    dt = np.dtype(_DTYPES[meta["dtype"]])
    shape = meta["shape"]
    return np.frombuffer(data[start:end], dtype=dt).reshape(shape)


_DTYPES = {
    "F64": "float64", "F32": "float32", "F16": "float16",
    "I64": "int64", "I32": "int32", "I16": "int16", "I8": "int8",
    "U8": "uint8", "BOOL": "bool",
}


def pack_row(qbytes, scales, n_in, group_size=128):
    """One output row -> TRQ1_0 packed bytes for the GGUF tensor.

    Strided-32 layout for SIMD: a 128-element group is stored as 32 bytes,
    byte m holding 4 base-3 digits for elements {m, m+32, m+64, m+96} of the
    group (3^4 = 81 < 256). Digit plane k pairs with a q8_K window of 32, so
    the vec_dot can use vpshufb/AVX2 maddubs like TQ1_0 does.
    ggml ne0 == n_in; blocks of 256 along ne0; each block = two 128-groups.

    Source qweight is the redux checkpoint's contiguous packing: element e of
    the row sits in byte e//5 at base-3 digit e%5 (5 digits/byte).
    """
    pow3_src = (1, 3, 9, 27, 81)   # source: 5 digits/byte
    pow3_dst = (1, 3, 9, 27)       # dest:   4 digits/byte
    n_groups = n_in // group_size
    n_blocks = n_in // 256
    buf = np.zeros(n_blocks * 68, dtype=np.uint8)
    for b in range(n_blocks):
        for g in range(2):
            gi = b * 2 + g
            if gi >= n_groups:
                break
            base = b * 68 + g * 32
            for m in range(32):
                byte = 0
                for k in range(4):
                    e = gi * group_size + m + k * 32   # element within the row
                    code = (qbytes[e // 5] // pow3_src[e % 5]) % 3
                    byte += int(code) * pow3_dst[k]
                buf[base + m] = byte
            scale = np.float16(float(scales[gi]))
            buf[b*68 + 64 + g*2 : b*68 + 66 + g*2] = np.frombuffer(
                np.array([scale], dtype=np.float16), dtype=np.uint8)
    return buf


def should_ternary(name, ternary_modules):
    """Name (sans .weight) -> True if this module is ternary (linear forms)."""
    return name in ternary_modules


def lift_featurizer(ref_gguf, d, w):
    """Copy the mel window + filterbank from a stock parakeet GGUF.

    The redux checkpoint ships no featurizer buffers (Photon derives mel at
    runtime); the C++ side needs the exact NeMo window/fb to stay bit-identical
    with the stock engine. They are architecture-identical, so lifting them
    from the reference TDT GGUF is exact rather than a re-derivation.
    """
    if ref_gguf is None:
        print("warning: no --featurizer-from; mel window/fb omitted (zeros)",
              file=__import__("sys").stderr)
        return
    import gguf as _gguf
    r = _gguf.GGUFReader(ref_gguf)
    for want in ("preprocessor.featurizer.window", "preprocessor.featurizer.fb"):
        t = next((t for t in r.tensors if t.name == want), None)
        if t is None:
            raise SystemExit(f"featurizer-ref missing {want}")
        raw = np.frombuffer(t.data, dtype=np.float32)
        if want.endswith("fb"):
            # file/ne order is [n_bins, n_mels, 1] (bins fastest, matching
            # parakeet.cpp's fb_[m*n_bins+b]); pass numpy C-order [1, n_mels,
            # n_bins] so the writer stores the same byte stream.
            n_mels, n_bins = 128, raw.size // 128
            arr = raw.reshape(1, n_mels, n_bins)
        else:
            arr = raw.reshape(-1)
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        w.add_tensor(want, arr)
    print(f"  featurizer lifted from {ref_gguf}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="redux checkpoint dir (config.json + model.safetensors + ternary.json + tokenizer.json)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--featurizer-from", default=None,
                    help="stock parakeet GGUF to lift the mel window/filterbank from (recommended: tdt-0.6b-v3-f16.gguf)")
    args = ap.parse_args()

    d = pathlib.Path(args.model_dir)
    cfg = json.load(open(d / "config.json"))
    ternary = json.load(open(d / "ternary.json"))
    header, data = read_safetensors(d / "model.safetensors")

    # which modules are ternary: linears pack to TRQ1_0; conv1d modules are
    # ternary in the checkpoint but must be dequantized to F32 for the engine
    # (they are reshaped [1,in,out] in-graph, which cannot survive
    # block-quantized storage -- same rule as the stock converter).
    ternary_names = set()
    conv_ternary = set()
    for m in ternary["quantized_modules"]:
        name = m["name"]
        if m.get("as_conv1d", False):
            conv_ternary.add(name)
        else:
            ternary_names.add(name)

    # ---- architecture metadata (same KV names the C++ loader reads) -------
    enc = cfg["encoder_config"]
    w = gguf.GGUFWriter(args.output, "parakeet")
    w.add_string("general.name", "moondream/parakeet-redux")
    # What this file IS, beyond the source name: the novel quant type, its packing
    # decision (unrecoverable from metadata otherwise) and the derivation. Stock
    # tools crash on type 42 rather than reporting it, so spell it out here.
    w.add_string("general.description",
                 "moondream/parakeet-redux encoder requantized to "
                 "GGML_TYPE_TRQ1_0 (ternary, 2.125 bpw, strided-32 packing); "
                 "conv/norms/featurizer kept F32. Requires ggml with type 42.")
    w.add_string("parakeet.quant.type",           "TRQ1_0")
    w.add_string("parakeet.quant.packing",        "strided-32")
    w.add_float32("parakeet.quant.bpw",           2.125)
    w.add_string("parakeet.quant.source_packing", "contiguous-5")
    w.add_string("parakeet.arch", "tdt")
    w.add_uint32("parakeet.encoder.feat_in", int(enc["num_mel_bins"]))
    w.add_uint32("parakeet.encoder.d_model", int(enc["hidden_size"]))
    w.add_uint32("parakeet.encoder.n_layers", int(enc["num_hidden_layers"]))
    w.add_uint32("parakeet.encoder.n_heads", int(enc["num_attention_heads"]))
    w.add_uint32("parakeet.encoder.ff_dim", int(enc["intermediate_size"]))
    w.add_uint32("parakeet.encoder.conv_kernel", int(enc["conv_kernel_size"]))
    w.add_string("parakeet.encoder.conv_norm_type", "batch_norm")
    w.add_uint32("parakeet.encoder.subsampling_factor", int(enc["subsampling_factor"]))
    w.add_uint32("parakeet.encoder.subsampling_conv_channels", int(enc["subsampling_conv_channels"]))
    w.add_bool("parakeet.encoder.xscaling", bool(enc.get("scale_input", False)) or False)
    w.add_uint32("parakeet.encoder.pos_emb_max_len", int(enc.get("max_position_embeddings", 5000)))
    w.add_bool("parakeet.encoder.use_bias", True)

    vocab = int(cfg["vocab_size"]) - 1    # 8193 -> 8192 real tokens; blank sits at 8192
    w.add_uint32("parakeet.vocab_size", vocab)
    w.add_uint32("parakeet.blank_id", int(cfg["blank_token_id"]))

    tok = json.load(open(d / "tokenizer.json"))
    vt = tok["model"]["vocab"]
    id_to_token = {int(v): str(k) for k, v in vt.items()}
    pieces = []
    for i in range(int(cfg["blank_token_id"])):  # 8192 real tokens
        pieces.append(id_to_token.get(i, f"<unk>"))
    w.add_array("parakeet.tokenizer.pieces", pieces)

    # transducer config: joint hidden == encoder_projector.out (640)
    w.add_uint32("parakeet.decoder.pred_hidden", int(cfg["decoder_hidden_size"]))
    w.add_uint32("parakeet.decoder.pred_rnn_layers", int(cfg["num_decoder_layers"]))
    w.add_uint32("parakeet.joint.joint_hidden", 640)
    w.add_string("parakeet.joint.activation", "relu")
    w.add_uint32("parakeet.decoding.max_symbols", int(cfg.get("max_symbols_per_step", 10)))
    durs = list(cfg["durations"])
    w.add_array("parakeet.tdt.durations", [int(x) for x in durs])

    # preprocessor (standard FastConformer mel config; the redux checkpoint
    # carries no featurizer buffers, so the C++ side computes its own mel)
    w.add_uint32("parakeet.preprocessor.sample_rate", 16000)
    w.add_uint32("parakeet.preprocessor.n_mels", int(enc["num_mel_bins"]))
    w.add_uint32("parakeet.preprocessor.n_fft", 512)
    w.add_uint32("parakeet.preprocessor.win_length", 400)
    w.add_uint32("parakeet.preprocessor.hop_length", 160)
    w.add_float32("parakeet.preprocessor.preemph", 0.97)
    w.add_float32("parakeet.preprocessor.mag_power", 2.0)
    w.add_string("parakeet.preprocessor.normalize", "per_feature")
    w.add_float32("parakeet.preprocessor.log_zero_guard", 2.0 ** -24)

    # ---- featurizer buffers (lifted from a stock GGUF) --------------------
    lift_featurizer(args.featurizer_from, d, w)

    # ---- tensors -----------------------------------------------------------
    written = ternary_written = dense_written = 0
    keys = [k for k in header if k != "__metadata__"]

    def _dequant(qarr, sarr):
        """Full dequant of a ternary module -> f32 [out, in] numpy."""
        out_rows, nbytes = qarr.shape
        n_groups = sarr.shape[1]
        n_in = n_groups * 128
        cols = np.arange(n_in)
        codes = (qarr[:, cols // 5].astype(np.int32) // (3 ** (cols % 5))) % 3
        return sarr[:, cols // 128].astype(np.float32) * (codes - 1)

    # qweight/scales pairs: linears -> TRQ1_0; conv1d -> dequantized F32
    pair_bases = sorted({k[:-len(".qweight")] for k in keys if k.endswith(".qweight")})
    for base in pair_bases:
        if base in conv_ternary:
            # dequantize to F32 (engine reshapes these; see converter notes)
            qarr = load_tensor(header, header[base + ".qweight"], data)
            sarr = load_tensor(header, header[base + ".scales"], data)
            dq = _dequant(qarr, sarr)
            ggml_ne = list(dq.shape[::-1])
            # conv pointwise is stored [out, in] in the GGUF and reshaped in-graph
            w.add_tensor(remap(base) + ".weight", np.ascontiguousarray(dq, dtype=np.float32))
            written += 1
            continue
        qm = header[base + ".qweight"]
        sm = header[base + ".scales"]
        qarr = load_tensor(header, qm, data)
        sarr = load_tensor(header, sm, data)
        out_rows, nbytes = qarr.shape
        n_groups = sarr.shape[1]
        n_in = n_groups * 128   # scale groups define the true in_features
        assert n_in % 256 == 0, f"{base}: in {n_in} not multiple of 256"

        rows = []
        for r in range(out_rows):
            rows.append(pack_row(qarr[r], sarr[r], n_in))
        packed = np.concatenate(rows)
        expected = (out_rows, (n_in // 256) * 68)
        packed = packed.reshape(expected)
        w.add_tensor(remap(base) + ".weight", packed,
                     raw_shape=packed.shape, raw_dtype=GGMLQuantizationType.TRQ1_0)
        ternary_written += 1
        written += 1

    # everything else: dense F32, names remapped to NeMo
    for name in keys:
        if name.endswith(".qweight") or name.endswith(".scales"):
            continue
        meta = header[name]
        if meta["shape"] == [] or (len(meta["shape"]) == 1 and meta["shape"][0] == 0):
            continue  # scalars
        base = name[:-len(".weight")] if name.endswith(".weight") else name
        if base in ternary_names or base in conv_ternary:
            continue  # already written as TRQ1_0 or dequantized F32
        if name.startswith("preprocessor.") and name not in (
                "preprocessor.featurizer.fb", "preprocessor.featurizer.window"):
            continue
        arr = load_tensor(header, meta, data)
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        arr = np.ascontiguousarray(arr)
        w.add_tensor(remap(name), arr)
        dense_written += 1
        written += 1

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {args.output}: arch=tdt vocab={vocab} "
          f"ternary={ternary_written} dense={dense_written} total={written}")


if __name__ == "__main__":
    main()
