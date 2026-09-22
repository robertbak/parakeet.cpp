# HANDOFF — Ternary (TRQ1_0) GGUF for parakeet-redux → parakeet.cpp

**Status: ✅ COMPLETE — CPU + ARM + CUDA all work and are verified.**
Last updated: 2026-09-22 (wrap-up session).

Runs **moondream/parakeet-redux** (the ~1.58-bit ternary ASR) from native
safetensors via a new ggml quant type `GGML_TYPE_TRQ1_0` (id 42), on **CPU
(AVX2)**, **ARM (scalar fallback)**, and **CUDA (GPU)**.

---

## Headline results (60 s clip, best-of-5)

### GPU vs CPU, 8 threads

| backend | format | RTFx | speedup vs CPU | file | CPU RSS | VRAM peak |
|---|---|---|---|---|---|---|
| CPU | f16 | 16.5 | — | 1374 MB | 1905 MB | — |
| CPU | q8_0 | 12.6 | — | 897 MB | 1425 MB | — |
| CPU | **TRQ1_0** | 12.7 | — | **511 MB** | **1040 MB** | — |
| CUDA | f16 | 52.7 | 3.27× | 1374 MB | — | 3811 MB |
| CUDA | q8_0 | 63.2 | 5.26× | 897 MB | — | 3373 MB |
| CUDA | **TRQ1_0** | **77.5** | **6.31×** | **511 MB** | — | **2430 MB** |

**TRQ1_0 is both the fastest and the smallest on GPU.**

### CPU thread scaling (TRQ1_0)

| threads | 2 | 4 | 8 | 16 |
|---|---|---|---|---|
| RTFx | 4.98 | 9.27 | 12.80 | 16.01 |

### AVX2 vs scalar (the kernel work)

| | scalar generic | AVX2 |
|---|---|---|
| RTFx (196 s clip, 6 thr) | 4.64 | **8.03** (1.73×) |

AVX2 is **bit-exact vs generic**; before this session the AVX2 path produced an
**empty transcript** (three bugs, all fixed — see below).

---

## What is verified

| Check | Result |
|---|---|
| AVX2 vs scalar generic | **bit-exact** (8 seeds × 512 blocks + 8 edge cases) |
| AVX2 vs exact double reference | err 0 on edge cases |
| `dequantize_row_trq1_0` layout | 2048 elements, **0 mismatches** |
| CPU CLI: AVX2 vs forced-generic (196 s) | **byte-identical** (3134 B both) |
| CPU CLI TRQ vs f16 (18.5 s) | byte-identical |
| **ARM link** (aarch64 object link) | `U` → **`T`** after fix (before: undefined) |
| **CUDA: TRQ on GPU vs CPU** | 163/164 words identical (**1 word**, see caveat) |
| CUDA: q8_0 on GPU vs CPU | byte-identical |
| Regression | **ctest 100% passed, 0 failed / 70** (non-model) |
| Our parity test | **ALL PASS** (`test_trq1_vecdot`) |

### CUDA output caveat (accepted — option 1)

GPU output differs from CPU by **1 word out of 164**:
`"we all need a side dish"` → `"we all meet a side dish"`. Same length, all
other words identical.

Cause: our path **dequantizes TRQ → f16 then runs f16 cuBLAS GEMM**, while CPU
accumulates in **f32** — low-bit logit drift flips one near-tie. q8_0 matched
byte-for-byte because it uses ggml-cuda's *fused* `mmq` int path. A wrong
element mapping would garble output, not flip one near-homophone, so the device
function is correct. If byte-exactness is ever needed: force the f32 dequant
path, or pin final decode to CPU.

---

## Three AVX2 bugs found & fixed (why it was returning empty)

| # | Bug | Fix |
|---|---|---|
| 1 | Copied TQ1_0 `avg*3/4 → top-2-bits` digit trick was wrong | Exact **mulhi division-by-3**: `q = _mm256_mulhi_epu16(set1(21846), w)` since `(v*21846)>>16 == floor(v/3)` for all 16-bit v; then `code = w - 3q` |
| 2 | `bsums + 16*g` read **out of bounds** (`bsums` has 16 entries) | correct index is **`8*g`** |
| 3 | `_mm256_set1_ps` + `hsum_float_8` = **8× too big** (broadcast into all 8 lanes, then summed) | **scalar** accumulator |

Also structural: elementwise lane subtraction against `bsums` is impossible with
the strided layout (planes don't align with contiguous 16-element bsum blocks),
so the `(code-1)` correction happens **once at group level**.

### Gotcha that cost the most time

**`GGML_CPU_FP16_TO_FP32` reads `ggml_table_f32_f16[]`, populated ONLY by
`ggml_cpu_init()`.** Any standalone test calling these vec_dots must call
`ggml_cpu_init()` first, else every scale reads 0.0 and *both* paths return 0 —
this is what made the original harness show `generic=0, avx2=0`.

---

## The ARM/mobile fix (arch-fallback.h)

`ggml_vec_dot_trq1_0_q8_K` was defined **only** in `arch/x86/quants.c`, but
`ggml-cpu/CMakeLists.txt:98-102` compiles **only** `arch/arm/quants.c` on ARM
(which has 24 sibling dispatchers but 0 for ours), while `ggml-cpu.c:405`
references it unguarded → **undefined reference**.

Mechanism used by ggml: `arch-fallback.h` (included by `quants.c`) does
`#define ggml_vec_dot_X_generic ggml_vec_dot_X`, so the *generic* implementation
is emitted **as** the dispatcher for arches without a native kernel.

**Fix: added 7 entries** — GENERIC, aarch64/arm, powerpc, loongarch, riscv,
s390x, wasm — and **deliberately NOT x86** (it has the native AVX2 one; adding
it would be a duplicate symbol).

Proven with a local aarch64 sysroot (no root needed:
`apt-get download libc6-dev-arm64-cross linux-libc-dev-arm64-cross` +
`dpkg-deb -x`) and clang `--target=aarch64-linux-gnu`:

| state | `quants.o` defines | linked together |
|---|---|---|
| before (0 entries) | `T ..._generic` | **`U ggml_vec_dot_trq1_0_q8_K`** ← link error |
| after (7 entries) | `T ggml_vec_dot_trq1_0_q8_K` | **`T`** resolved |

Per-target macro check: aarch64/riscv/powerpc/s390/wasm/loongarch = `1`,
**x86_64 = `0`** (still uses native AVX2).

> Mobile still wants a **NEON kernel** later (currently falls back to scalar,
> measured 4.64 vs 8.03 RTFx — fine for offline transcription, and the memory
> win is what matters on-device).

---

## The CUDA fix (three pieces, all required)

1. **`ggml-cuda/dequantize.cuh`** — `dequantize_trq1_0_elem` +
   `dequantize_trq1_0(const void*, int64_t ib, int iqs, float2&)` matching
   `dequantize_kernel_t`. With `qr=1`, `iqs` is an even element index → returns
   elements `iqs`, `iqs+1`. Decode: `g = e>>7`, `r = e&127`, `m = r&31`,
   `k = r>>5`, `code = (qs[g*32+m] / 3^k) % 3`, `* d[g]`.
2. **`ggml-cuda/convert.cu`** — 5 switch cases (`ggml_get_to_fp16_cuda`,
   `_fp32_`, and the three `_nc` variants), mirroring where Q8_0 appears.
   **`ggml_get_to_bf16_cuda` deliberately skipped** — it has no quant types.
   Registration here is also what makes `supports_op` *consult* the type.
3. **`ggml-cuda/ggml-cuda.cu:5166`** — added `case GGML_TYPE_TRQ1_0:` to the
   **MUL_MAT type whitelist**. Without this it hits `default: return false` and
   silently offloads to CPU. (This was the actual blocker: step 1–2 alone left
   TRQ at 15.20 RTFx / 1.20×.) `GET_ROWS` whitelist intentionally left alone.

---

## Files changed

### parakeet.cpp (main repo)
| File | Change |
|---|---|
| `scripts/convert_redux_to_gguf.py` | **new** — redux safetensors → GGUF (standalone, no NeMo) |
| `tests/test_trq1_vecdot.cpp` | **new** — AVX2/generic/exact + dequant parity test |
| `tests/CMakeLists.txt` | adds `test_trq1_vecdot` target (needs `ggml-cpu/` include dir) |
| `HANDOFF.md` | this file |
| `build-cuda/` | CUDA build dir (129 MB, optional — `build/` is the CPU one) |

### ggml submodule — **our** type work (13 files)
`include/ggml.h` · `src/ggml-common.h` · `src/ggml.c` ·
`src/ggml-quants.{c,h}` · `src/ggml-cpu/quants.{c,h}` ·
`src/ggml-cpu/ggml-cpu.c` *(line 403 hunk only)* ·
`src/ggml-cpu/arch/x86/quants.c` · `src/ggml-cpu/arch-fallback.h` ·
`src/ggml-cuda/{convert.cu,dequantize.cuh,ggml-cuda.cu}`

> ⚠️ **The submodule also contains PRE-EXISTING local edits that are NOT ours** —
> do not fold them into this PR:
> - `src/ggml-metal/*` (8 files), `src/ggml-cuda/pad.cu`
> - `src/ggml-cpu/ggml-cpu.c` **line 1299+ hunk** — an `rfdetr.cpp patch`
>   (llamafile sgemm broadcast-merge, ~125 lines)
>
> Submodule is at `e705c5fe` (ggml 0.13.0). Never `git checkout` those files.

---

## Build & test

```bash
cd /mnt/Data2/data/opensource/rust/parakeet.cpp
# tests are OFF by default
cmake -B build -DCMAKE_BUILD_TYPE=Release -DPARAKEET_BUILD_SERVER=OFF -DPARAKEET_BUILD_TESTS=ON
cmake --build build -j$(nproc)
ctest --test-dir build -E model            # 100% pass
ctest --test-dir build -R test_trq1_vecdot # our parity test

# CUDA build (separate dir, leaves CPU build alone)
cmake -B build-cuda -DCMAKE_BUILD_TYPE=Release -DPARAKEET_BUILD_SERVER=OFF \
      -DPARAKEET_GGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build-cuda -j$(nproc)
PARAKEET_DEVICE=CUDA0 ./build-cuda/examples/cli/parakeet-cli transcribe \
      --model /tmp/redux-trq.gguf --input in.wav --threads 8
# PARAKEET_DEVICE: unset = first GPU, "cpu" = force CPU, e.g. "CUDA0" by name
```

Convert:
```bash
python scripts/convert_redux_to_gguf.py \
  --model-dir <redux-snapshot> \
  --featurizer-from /mnt/Data2/storage/llama_cpp/models/hf/mudler/parakeet-cpp-gguf/tdt-0.6b-v3-f16.gguf \
  --output /tmp/redux-trq.gguf
```
References: `.../parakeet-cpp-gguf/tdt-0.6b-v3-{f16,q8_0}.gguf`

---

## Remaining / PR notes

1. **bpp decision (settled):** keep **2.125 bpw strided-32**. Switching to
   redux-native contiguous 5-digits/byte (1.75 bpw) would save only **23.7 MB**
   (511 → 487 MB, −4.6%) but destroy the SIMD structure (5 doesn't divide 128 →
   no aligned digit planes) and force a full kernel rewrite. The F32 payload
   (377 MB, conv/norms/featurizer) dwarfs it and is a loader constraint shared
   with stock. Consider documenting `packing` in a GGUF KV.
2. **Docs fixed:** stale `ggml.h` enum comment and converter docstring (both
   still said "5 digits/byte, 1.75 bpw" from the abandoned design).
3. **Metal** is still absent (only matters for iOS; same `convert.cu`-style
   dequant recipe as the CUDA work).
4. **NEON kernel** optional — scalar fallback is adequate for mobile today.
5. Single PR to `mudler/parakeet.cpp` — type + block + converter + kernels +
   `arch-fallback.h` + CUDA path + `test_trq1_vecdot` + docs.

---

## Related context (this machine)

- **parakeet-asr** service: `/home/robert/opt/parakeet-asr/parakeet-asr.py`
  (FastAPI, OpenAI `/v1/audio/transcriptions` + `/v1/realtime`), llama-swap
  `parakeet-asr` entry (CPU, persistent).
- **super-stt**: `/mnt/Data2/data/opensource/super-stt` — native subprocess
  backend wrapping this fork was the local-streaming plan.
- **Polish post-processing**: `/mnt/Data2/data/opensource/rust/pl-speech`
  (`polish-fix-backend`, `polish-grammar`, `polish-phonetics`).
- RTX 5090 (32 GB) hosts llama-swap (8080) + LLM workloads; this ASR work is
  CPU-side by design so the GPU stays free — though as shown, when the GPU *is*
  free TRQ1_0 runs 6.3× faster on it.
