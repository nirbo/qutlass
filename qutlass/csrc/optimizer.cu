#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "include/common.h"

namespace QUTLASS {

__device__ const float fp4_grid[] = {
    -6.0f, -4.0f, -3.0f, -2.0f, -1.5f, -1.0f, -0.5f, -0.0f,
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f
};

__device__ const int fp4_codes_raw[] = {
    15, 14, 13, 12, 11, 10, 9, 8, 0, 1, 2, 3, 4, 5, 6, 7
};

__device__ const float fp4_vals[] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

__device__ inline uint8_t quantize_sr_fp4(float x, curandState_t* state) {
    if (x <= fp4_grid[0]) return 15;
    if (x >= fp4_grid[15]) return 7;

    int i = 0, j = 15;
    while (i <= j) {
        int mid = (i + j) / 2;
        if (fp4_grid[mid] < x) i = mid + 1;
        else j = mid - 1;
    }
    int hi_idx = i;
    int lo_idx = i - 1;

    float prob_hi = (x - fp4_grid[lo_idx]) / (fp4_grid[hi_idx] - fp4_grid[lo_idx] + 1e-8f);
    int bucket_idx = (curand_uniform(state) < prob_hi) ? hi_idx : lo_idx;
    return (uint8_t)(fp4_codes_raw[bucket_idx] & 0xF);
}

__global__ void fused_adamw_fp4_update_kernel(
    uint8_t* values, float* scales, float* m, float* v, const float* grad,
    float lr, float beta1, float beta2, float eps, float weight_decay,
    int step, float gscale, float unscale, int M, int K, int SF, unsigned long long seed) {

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int group = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= M || group >= K / SF) return;

    curandState_t state;
    curand_init(seed, row * (K / SF) + group, 0, &state);

    float bc1 = 1.0f - powf(beta1, (float)step);
    float bc2 = 1.0f - powf(beta2, (float)step);

    float group_codes[32], group_updates[32];
    float current_scale = fmaxf(scales[row * (K / SF) + group], 1e-8f);
    float norm_v2 = 1e-8f, dot_uv = 0.0f;

    for (int i = 0; i < SF; ++i) {
        int col = group * SF + i;
        int idx = row * K + col;
        uint8_t code = (col % 2 == 0) ? (values[row * (K / 2) + col / 2] & 0xF) : (values[row * (K / 2) + col / 2] >> 4);
        group_codes[i] = fp4_vals[code];

        float g = grad[idx] * unscale;
        float m_val = m[idx] * beta1 + g * (1.0f - beta1);
        float v_val = v[idx] * beta2 + g * g * (1.0f - beta2);
        m[idx] = m_val; v[idx] = v_val;

        float update = (m_val / bc1) / (sqrtf(v_val / bc2) + eps);
        group_updates[i] = update;
        norm_v2 += group_codes[i] * group_codes[i];
        dot_uv += update * group_codes[i];
    }

    float alpha = dot_uv / norm_v2;
    for (int i = 0; i < SF; i += 2) {
        float nv1 = group_codes[i] - lr * (group_updates[i] - alpha * group_codes[i]) * (gscale / current_scale);
        float nv2 = group_codes[i+1] - lr * (group_updates[i+1] - alpha * group_codes[i+1]) * (gscale / current_scale);
        values[row * (K / 2) + (group * SF + i) / 2] = (quantize_sr_fp4(nv1, &state) & 0xF) | (quantize_sr_fp4(nv2, &state) << 4);
    }

    if (weight_decay != 0.0f) current_scale *= (1.0f - lr * weight_decay);
    scales[row * (K / SF) + group] = fmaxf(current_scale - lr * gscale * alpha, 1e-8f);
}

void fused_adamw_fp4_update_host(
    torch::Tensor& values,
    torch::Tensor& scales,
    torch::Tensor& m,
    torch::Tensor& v,
    const torch::Tensor& grad,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    int step,
    float gscale,
    float unscale,
    int SF,
    unsigned long long seed)
{
    int M = values.size(0);
    int K = values.size(1) * 2;
    dim3 threads(32, 1);
    dim3 blocks((M + 31) / 32, K / SF);

    fused_adamw_fp4_update_kernel<<<blocks, threads>>>(
        (uint8_t*)values.data_ptr(),
        (float*)scales.data_ptr(),
        (float*)m.data_ptr(),
        (float*)v.data_ptr(),
        (const float*)grad.data_ptr(),
        lr, beta1, beta2, eps, weight_decay,
        step, gscale, unscale, M, K, SF, seed
    );
}

} // namespace QUTLASS
