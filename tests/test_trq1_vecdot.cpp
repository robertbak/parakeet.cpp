// Compare generic vs AVX2 trq1_0 vec_dot on random data.
//
// Verifies, over many random blocks:
//   1. AVX2  == generic   bit-exact (same int sums, same float reduction)
//   2. generic == exact double-precision reference (within float rounding)
//
// Also covers adversarial edge cases (all-zero bytes, max digits, q8 extremes).
#include <cstdio>
#include <vector>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <random>
#include "ggml-cpu/quants.h"
#include "ggml-quants.h"   // dequantize_row_trq1_0 (the engine's to_float path)
#include "ggml-cpu.h"

static const uint8_t pow3[4] = { 1, 3, 9, 27 };

// Exact double-precision reference: dequantize both sides, dot product.
static double ref_dot(const block_trq1_0 * x, const block_q8_K * y, int nb) {
    double total = 0.0;
    for (int i = 0; i < nb; ++i) {
        for (int g = 0; g < 2; ++g) {
            const double d = (double)ggml_fp16_to_fp32(x[i].d[g]) * (double)y[i].d;
            double sum = 0.0;
            for (int m = 0; m < 32; ++m) {
                for (int k = 0; k < 4; ++k) {
                    const int code = (x[i].qs[g*32 + m] / pow3[k]) % 3;
                    sum += (double)(code - 1) * (double)y[i].qs[g*128 + m + k*32];
                }
            }
            total += d * sum;
        }
    }
    return total;
}

static void fill_random(block_trq1_0 * x, block_q8_K * y, int nb, std::mt19937 &rng) {
    for (int i = 0; i < nb; ++i) {
        for (int g = 0; g < 2; ++g) {
            for (int m = 0; m < 32; ++m) {
                uint8_t b = 0;
                for (int k = 0; k < 4; ++k) b += (uint8_t)(rng() % 3) * pow3[k];
                x[i].qs[g*32 + m] = b;
            }
            x[i].d[g] = ggml_fp32_to_fp16(0.5f + (float)(rng() % 1000) / 1000.0f);
        }
        int sum[16] = { 0 };
        for (int j = 0; j < QK_K; ++j) {
            y[i].qs[j] = (int8_t)(rng() % 256 - 128);
            sum[j / 16] += y[i].qs[j];
        }
        y[i].d = 0.75f;   // block_q8_K::d is a float (delta), not fp16
        for (int j = 0; j < 16; ++j) y[i].bsums[j] = (int16_t)sum[j];
    }
}

int main() {
    // GGML_CPU_FP16_TO_FP32 reads ggml_table_f32_f16[] which is only populated
    // by ggml_cpu_init(). Without this every scale reads back as 0.0.
    ggml_cpu_init();

    int failures = 0;
    double worst_generic = 0.0, worst_avx2 = 0.0;
    int bitexact_avx2 = 1;

    // --- 1. many random blocks, several seeds ---
    for (unsigned seed = 1; seed <= 8; ++seed) {
        const int nb = 512;
        const int n  = nb * QK_K;
        std::mt19937 rng(seed);

        std::vector<block_trq1_0> x(nb);
        std::vector<block_q8_K>   y(nb);
        fill_random(x.data(), y.data(), nb, rng);

        float avx2 = 0, generic = 0;
        ggml_vec_dot_trq1_0_q8_K       (n, &avx2,    0, x.data(), 0, y.data(), 0, 1);
        ggml_vec_dot_trq1_0_q8_K_generic(n, &generic, 0, x.data(), 0, y.data(), 0, 1);

        const double exact = ref_dot(x.data(), y.data(), nb);

        if (memcmp(&avx2, &generic, sizeof(float)) != 0) bitexact_avx2 = 0;
        worst_generic = std::max(worst_generic, fabs(exact - (double)generic));
        worst_avx2    = std::max(worst_avx2,    fabs(exact - (double)avx2));
    }

    printf("random  : worst|generic-exact| = %.6g   worst|avx2-exact| = %.6g   avx2==generic bit-exact: %s\n",
           worst_generic, worst_avx2, bitexact_avx2 ? "YES" : "NO");
    if (!bitexact_avx2) failures++;

    // --- 2. adversarial edge cases ---
    struct Case { const char *name; int qs_fill; int8_t q8_fill; float xd; float yd; };
    const Case cases[] = {
        { "all-zero digits / zero q8", 0x00,   0,   1.0f,  1.0f },
        { "all-zero digits / q8=+127", 0x00,   127, 1.0f,  1.0f },
        { "all-zero digits / q8=-128", 0x00,  -128, 1.0f,  1.0f },
        { "max digit 80 (all code=2)", 80,     0,   1.0f,  1.0f },
        { "max digit 80 / q8=+127",    80,     127, 1.0f,  1.0f },
        { "max digit 80 / q8=-128",    80,    -128, 1.0f,  1.0f },
        { "byte 255 (digit overflow)", 255,    127, 0.5f,  2.0f },
        { "alt digits (0x2A=base3 101)",0x2A,  -1,   1.5f,  0.25f },
    };

    for (const auto &c : cases) {
        const int nb = 4;
        const int n  = nb * QK_K;
        std::vector<block_trq1_0> x(nb);
        std::vector<block_q8_K>   y(nb);

        for (int i = 0; i < nb; ++i) {
            memset(x[i].qs, c.qs_fill, sizeof(x[i].qs));
            x[i].d[0] = ggml_fp32_to_fp16(c.xd);
            x[i].d[1] = ggml_fp32_to_fp16(c.xd);
            int sum[16] = { 0 };
            for (int j = 0; j < QK_K; ++j) {
                y[i].qs[j] = c.q8_fill == -1 ? (int8_t)((j * 7) % 256 - 128) : c.q8_fill;
                sum[j / 16] += y[i].qs[j];
            }
            y[i].d = c.yd;
            for (int j = 0; j < 16; ++j) y[i].bsums[j] = (int16_t)sum[j];
        }

        float avx2 = 0, generic = 0;
        ggml_vec_dot_trq1_0_q8_K       (n, &avx2,    0, x.data(), 0, y.data(), 0, 1);
        ggml_vec_dot_trq1_0_q8_K_generic(n, &generic, 0, x.data(), 0, y.data(), 0, 1);
        const double exact = ref_dot(x.data(), y.data(), nb);

        const bool bit_ok = memcmp(&avx2, &generic, sizeof(float)) == 0;
        const double err  = fabs(exact - (double)avx2);
        const double tol  = 1e-4 * fmax(1.0, fabs(exact));
        const bool ok = bit_ok && err <= tol;
        if (!ok) failures++;

        printf("  [%s] %-26s exact=%12.5f avx2=%12.5f generic=%12.5f  bit=%s err=%.3g\n",
               ok ? "PASS" : "FAIL", c.name, exact, avx2, generic,
               bit_ok ? "eq" : "NE", err);
    }

    // --- 3. dequantize_row_trq1_0 must agree with what vec_dot assumes ---
    // element g*128 + m + k*32 == (code-1) * scale_g
    {
        const int nb = 8;
        std::mt19937 rng(99);
        std::vector<block_trq1_0> dx(nb);
        for (int i = 0; i < nb; ++i) {
            for (int g = 0; g < 2; ++g) {
                for (int m = 0; m < 32; ++m) {
                    uint8_t b = 0;
                    for (int k = 0; k < 4; ++k) b += (uint8_t)(rng() % 3) * pow3[k];
                    dx[i].qs[g*32 + m] = b;
                }
                dx[i].d[g] = ggml_fp32_to_fp16(0.25f + (float)(rng() % 800) / 1000.0f);
            }
        }
        std::vector<float> out((size_t)nb * QK_K, -999.0f);
        dequantize_row_trq1_0(dx.data(), out.data(), (int64_t)nb * QK_K);

        int bad = 0;
        for (int i = 0; i < nb; ++i) {
            for (int g = 0; g < 2; ++g) {
                const float d = ggml_fp16_to_fp32(dx[i].d[g]);
                for (int m = 0; m < 32; ++m) {
                    for (int k = 0; k < 4; ++k) {
                        const int code = (dx[i].qs[g*32 + m] / pow3[k]) % 3;
                        const float want = (float)(code - 1) * d;
                        const float got  = out[(size_t)i * QK_K + g*128 + m + k*32];
                        if (memcmp(&want, &got, sizeof(float)) != 0) bad++;
                    }
                }
            }
        }
        const bool ok = bad == 0;
        if (!ok) failures++;
        printf("  [%s] dequantize_row_trq1_0 layout   elements_checked=%d mismatches=%d\n",
               ok ? "PASS" : "FAIL", nb * QK_K, bad);
    }

    printf("%s\n", failures == 0 ? "ALL PASS" : "FAILURES");
    return failures == 0 ? 0 : 1;
}
