#ifndef CUTIE_ORT_CORE_CUDA_KERNELS_H
#define CUTIE_ORT_CORE_CUDA_KERNELS_H

#include <cstdint>

/// @file cuda_kernels.h
/// GPU 张量操作的 CUDA kernel 声明。
/// 所有函数在默认 CUDA stream 上执行（stream = 0）。

namespace cutie
{
namespace cuda
{

/// 沿最后一个维度拼接两个张量。
/// a: [outer, a_inner],  b: [outer, b_inner]  →  out: [outer, a_inner + b_inner]
void concat_last_dim(const float* a, int64_t a_inner, const float* b, int64_t b_inner,
                     float* out, int64_t outer);

/// 沿最后一个维度切片：src [outer, src_inner] → dst [outer, slice_len]
/// 从 offset 开始取 slice_len 个元素。
void slice_last_dim(const float* src, int64_t src_inner, float* dst, int64_t slice_len,
                    int64_t offset, int64_t outer);

/// Sigmoid 激活：out[i] = 1 / (1 + exp(-x[i]))
void sigmoid(const float* x, float* out, int64_t n);

/// 聚合（aggregate）：prob_no_bg [num_obj, HW] → logits [num_obj+1, HW]
/// 计算 bg = prod(1 - prob)，拼接 [bg, prob]，转为 logits（不做 softmax）
void aggregate_logits(const float* prob_no_bg, float* out, int num_obj, int hw);

/// 聚合 + Softmax（aggregate_with_bg）。
/// prob_no_bg: [num_obj, H, W]  →  out: [num_obj+1, H, W]
/// 先将 prob 转为 logit，添加 bg 通道（logit=0），再做 softmax。
void aggregate_softmax(const float* prob_no_bg, float* out, int num_obj, int hw);

/// Softmax 沿通道维度：logits [C, HW] → prob [C, HW]
void softmax_channels(const float* logits, float* out, int C, int hw);

/// 各向异性 L2 相似度计算。
/// memory_key: [B, CK, N],  memory_shrinkage: [B, 1, N]
/// query_key:  [B, CK, HW], query_selection:  [B, CK, HW]
/// out: [B, N, HW]
/// similarity = sum_c( (mk_c * ms) * (qk_c * qs_c) ) 即带权内积
void get_similarity(const float* memory_key, const float* memory_shrinkage,
                    const float* query_key, const float* query_selection, float* out, int B,
                    int CK, int N, int HW);

/// Top-K 稀疏 Softmax。
/// similarity: [B, N, HW] → affinity: [B, N, HW]
/// 对每个 (b, hw) 位置，只保留 top_k 个最大值做 softmax，其余置零。
/// 同时输出 usage: [B, N]（每个 memory token 被选中的次数之和）。
/// usage 可为 nullptr（不需要时跳过）。
void top_k_softmax(const float* similarity, float* affinity, float* usage, int B, int N, int HW,
                   int top_k);

/// idx mask → one-hot 编码。
/// mask: [H*W] int32,  objects: [num_obj] int32
/// out: [num_obj, H*W] float32
void one_hot_encode(const int32_t* mask, const int32_t* objects, float* out, int num_obj, int hw);

/// 合并预测 mask 和输入 mask。
/// pred_no_bg: [num_existing, HW],  input_mask: [HW] int32
/// 将 input_mask > 0 的位置在 pred_no_bg 中置零。
void mask_merge_zero(float* pred_no_bg, const int32_t* input_mask, int num_existing, int hw);

/// GPU 上的 memset 为零。
void fill_zero(float* ptr, int64_t n);

/// GPU 上的 memcpy（device to device）。
void copy_d2d(float* dst, const float* src, int64_t n);

/// GPU 上的逐元素加法：dst[i] += src[i]
void add_inplace(float* dst, const float* src, int64_t n);

/// GPU 双线性插值 resize。
/// src: [channels, src_h, src_w],  dst: [channels, dst_h, dst_w]
/// 直接操作连续 GPU 内存，不依赖 CUDA Texture Object，无 pitch 对齐要求。
void bilinear_resize(const float* src, int src_h, int src_w, float* dst, int dst_h, int dst_w,
                     int channels);

}  // namespace cuda
}  // namespace cutie

#endif  // CUTIE_ORT_CORE_CUDA_KERNELS_H
