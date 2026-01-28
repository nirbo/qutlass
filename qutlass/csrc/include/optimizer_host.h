#pragma once
#include <ATen/ATen.h>

namespace QUTLASS {

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
    unsigned long long seed);

} // namespace QUTLASS
