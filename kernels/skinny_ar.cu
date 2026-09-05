// skinny_ar: two-rank fp16 all-reduce through pinned host memory, for boxes
// without GPU P2P (KVM guests, mismatched boards). NCCL's LL protocol costs
// ~44 us per call here for the 40-160 KB decode all-reduces and the fork's
// custom all-reduce needs P2P/IPC, so neither helps. This does the whole
// exchange inside ONE kernel: each block writes its chunk to a host slot,
// fences, raises a per-block flag, spins on the peer's flag, then reads the
// peer's chunk straight from host memory and adds it in place. No CPU work,
// no stream sync, capturable in CUDA graphs (the call counter lives in device
// memory and is advanced by the last block to finish).
//
// Numerics: out = mine + peer in fp16, a single rounding, commutative, so the
// two ranks produce identical bits and match NCCL's 2-rank f16 sum.
//
// Slot reuse: double buffering by call parity. Before call n+2 overwrites
// slot (n & 1), this rank has already observed the peer's flag for call n+1,
// which the peer raises only after it finished call n on its own stream.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstring>
#include <string>

namespace {

constexpr int kMaxBlocks = 16;
constexpr int kMaxChunks = 64;   // 4 KB chunks -> 256 KB max message
constexpr int kThreads = 256;
constexpr int kFlagStride = 32;  // uints; 128 B per flag, one cache line each
// flags region: [parity 2][rank 2][kMaxChunks] * kFlagStride uints (kMaxChunks >= kMaxBlocks)
constexpr size_t kFlagsBytes = 2ull * 2 * kMaxChunks * kFlagStride * sizeof(unsigned);

struct State {
  bool ready = false;
  int rank = -1;
  int world = 0;
  size_t slot_bytes = 0;     // per (parity, rank) data slot
  size_t map_bytes = 0;
  void* host = nullptr;      // mmap base
  void* dev = nullptr;       // device pointer to the same memory
  unsigned* seq_dev = nullptr;
  unsigned* done_dev = nullptr;
  std::string path;
};
State g;

__device__ __forceinline__ int4 ld_volatile_int4(const int4* p) {
  int4 v;
  asm volatile("ld.volatile.global.v4.s32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

__device__ __forceinline__ void st_volatile_int4(int4* p, int4 v) {
  asm volatile("st.volatile.global.v4.s32 [%0], {%1,%2,%3,%4};"
               :: "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}

__device__ __forceinline__ int4 add_half8(int4 a, int4 b) {
  const half2* pa = reinterpret_cast<const half2*>(&a);
  const half2* pb = reinterpret_cast<const half2*>(&b);
  int4 o;
  half2* po = reinterpret_cast<half2*>(&o);
#pragma unroll
  for (int i = 0; i < 4; i++) po[i] = __hadd2(pa[i], pb[i]);
  return o;
}

__device__ __forceinline__ int4 ld_cg_int4(const int4* p) {
  int4 v;
  asm volatile("ld.global.cg.v4.s32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

// MODE 0: volatile stores + volatile loads. MODE 1: plain stores + volatile
// loads. MODE 2: plain stores + ld.global.cg loads (relies on system memory
// not being cached in L2; verified by the stress test before use).
// x -> out: [n8] int4 (8 halves each); out may alias x.
template <int MODE>
__global__ void __launch_bounds__(kThreads)
skinny_ar2_kernel(const int4* __restrict__ x, int4* __restrict__ out, int n8, unsigned* flags,
                  int4* data, int rank, int slot_int4, unsigned* seq_dev, unsigned* done_dev) {
  __shared__ unsigned s_seq;
  if (threadIdx.x == 0) s_seq = *reinterpret_cast<volatile unsigned*>(seq_dev) + 1u;
  __syncthreads();
  const unsigned seq = s_seq;
  const int par = seq & 1u;
  const int peer = rank ^ 1;
  int4* my_slot = data + (size_t)(par * 2 + rank) * slot_int4;
  const int4* peer_slot = data + (size_t)(par * 2 + peer) * slot_int4;
  volatile unsigned* my_flag =
      flags + ((size_t)(par * 2 + rank) * kMaxChunks + blockIdx.x) * kFlagStride;
  volatile unsigned* peer_flag =
      flags + ((size_t)(par * 2 + peer) * kMaxChunks + blockIdx.x) * kFlagStride;

  const int per_block = (n8 + gridDim.x - 1) / gridDim.x;
  const int beg = blockIdx.x * per_block;
  const int end = min(n8, beg + per_block);

  // 1. publish my chunk (posted PCIe writes), then fence + flag.
  for (int i = beg + threadIdx.x; i < end; i += kThreads) {
    if (MODE == 0) st_volatile_int4(my_slot + i, x[i]);
    else my_slot[i] = x[i];
  }
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) {
    *my_flag = seq;
    // 2. wait for the peer's chunk.
    while (*peer_flag != seq) { }
  }
  __syncthreads();
  // 3. read the peer's chunk from host memory and add in place.
  for (int i = beg + threadIdx.x; i < end; i += kThreads) {
    const int4 p = (MODE == 2) ? ld_cg_int4(peer_slot + i) : ld_volatile_int4(peer_slot + i);
    out[i] = add_half8(x[i], p);
  }
  // 4. last block to finish advances the call counter.
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    const unsigned d = atomicAdd(done_dev, 1u);
    if (d == gridDim.x - 1) {
      *done_dev = 0;
      __threadfence();
      *reinterpret_cast<volatile unsigned*>(seq_dev) = seq;
    }
  }
}

// Fused 2-rank all-reduce + residual add + Gemma RMSNorm (x * (1 + w)),
// the fork's `sm70_tp2_all_reduce_gemma_rms_norm` contract: inp fp16 [M,H],
// residual fp16 or fp32 [M,H], weight fp16 or fp32 [H] -> (normalized fp16
// [M,H], residual_out fp32 [M,H]). One block per row: the row is exchanged
// through host memory exactly like skinny_ar2_kernel (same fp16 sum), then
// r = float(sum) + float(residual); var = mean(r^2); y = r * rsqrt(var+eps)
// * (1 + w). H must be a multiple of 8 and <= kNormThreads * 8 * 4.
constexpr int kNormThreads = 256;

__device__ __forceinline__ float to_f32(float v) { return v; }
__device__ __forceinline__ float to_f32(half v) { return __half2float(v); }

template <typename TRes, typename TW>
__global__ void __launch_bounds__(kNormThreads)
skinny_ar2_gemma_norm_kernel(const int4* __restrict__ x, const TRes* __restrict__ residual,
                             const TW* __restrict__ weight, half* __restrict__ out,
                             float* __restrict__ res_out, int H, float eps,
                             unsigned* flags, int4* data, int rank, int slot_int4,
                             unsigned* seq_dev, unsigned* done_dev) {
  __shared__ unsigned s_seq;
  __shared__ float s_red[kNormThreads / 32];
  if (threadIdx.x == 0) s_seq = *reinterpret_cast<volatile unsigned*>(seq_dev) + 1u;
  __syncthreads();
  const unsigned seq = s_seq;
  const int par = seq & 1u;
  const int peer = rank ^ 1;
  const int row = blockIdx.x;
  const int H8 = H >> 3;
  int4* my_slot = data + (size_t)(par * 2 + rank) * slot_int4;
  const int4* peer_slot = data + (size_t)(par * 2 + peer) * slot_int4;
  volatile unsigned* my_flag = flags + ((size_t)(par * 2 + rank) * kMaxChunks + row) * kFlagStride;
  volatile unsigned* peer_flag = flags + ((size_t)(par * 2 + peer) * kMaxChunks + row) * kFlagStride;
  const int4* xrow = x + (size_t)row * H8;

  // 1. publish my row
  for (int i = threadIdx.x; i < H8; i += kNormThreads) my_slot[(size_t)row * H8 + i] = xrow[i];
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) {
    *my_flag = seq;
    while (*peer_flag != seq) { }
  }
  __syncthreads();
  // 2. sum + residual (fp32), accumulate sum of squares; keep r in registers
  //    (up to 4 int4 = 32 elements per thread).
  float r[32];
  float ss = 0.f;
  int cnt = 0;
#pragma unroll
  for (int it = 0; it < 4; it++) {
    const int i = threadIdx.x + it * kNormThreads;
    if (i < H8) {
      const int4 mine = xrow[i];
      const int4 p = ld_volatile_int4(peer_slot + (size_t)row * H8 + i);
      const int4 sum = add_half8(mine, p);
      const half* sh = reinterpret_cast<const half*>(&sum);
#pragma unroll
      for (int j = 0; j < 8; j++) {
        const float rv = __half2float(sh[j]) + to_f32(residual[(size_t)row * H + i * 8 + j]);
        r[it * 8 + j] = rv;
        ss = fmaf(rv, rv, ss);
      }
      cnt++;
    }
  }
  // 3. block reduce sum of squares
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
  if ((threadIdx.x & 31) == 0) s_red[threadIdx.x >> 5] = ss;
  __syncthreads();
  if (threadIdx.x < 32) {
    float v = (threadIdx.x < kNormThreads / 32) ? s_red[threadIdx.x] : 0.f;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (threadIdx.x == 0) s_red[0] = v;
  }
  __syncthreads();
  const float inv = rsqrtf(s_red[0] / (float)H + eps);
  // 4. write normalized (fp16) and residual (fp32)
#pragma unroll
  for (int it = 0; it < 4; it++) {
    const int i = threadIdx.x + it * kNormThreads;
    if (i < H8) {
      __align__(16) half o8[8];
      __align__(16) float f8[8];
#pragma unroll
      for (int j = 0; j < 8; j++) {
        const float rv = r[it * 8 + j];
        const float w = 1.0f + to_f32(weight[i * 8 + j]);
        o8[j] = __float2half(rv * inv * w);
        f8[j] = rv;
      }
      *reinterpret_cast<int4*>(out + (size_t)row * H + i * 8) = *reinterpret_cast<const int4*>(o8);
      float4* fo = reinterpret_cast<float4*>(res_out + (size_t)row * H + i * 8);
      fo[0] = *reinterpret_cast<const float4*>(&f8[0]);
      fo[1] = *reinterpret_cast<const float4*>(&f8[4]);
    }
  }
  // 5. last block advances the call counter
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    const unsigned d = atomicAdd(done_dev, 1u);
    if (d == gridDim.x - 1) {
      *done_dev = 0;
      __threadfence();
      *reinterpret_cast<volatile unsigned*>(seq_dev) = seq;
    }
  }
}

// Diagnostics: split the exchange into its phases. Not used in serving.
__global__ void skinny_ar_diag_write(const int4* __restrict__ x, int n8, int4* slot) {
  for (int i = blockIdx.x * kThreads + threadIdx.x; i < n8; i += gridDim.x * kThreads) slot[i] = x[i];
  __threadfence_system();
}
__global__ void skinny_ar_diag_read(int4* __restrict__ x, int n8, const int4* slot) {
  for (int i = blockIdx.x * kThreads + threadIdx.x; i < n8; i += gridDim.x * kThreads)
    x[i] = add_half8(x[i], ld_volatile_int4(slot + i));
}

}  // namespace

void skinny_ar_diag(torch::Tensor x, int64_t which, int64_t blocks) {
  TORCH_CHECK(g.ready && x.is_cuda() && x.dtype() == torch::kFloat16 && x.is_contiguous());
  const int n8 = (int)(x.numel() * 2 / 16);
  int4* slot = reinterpret_cast<int4*>(reinterpret_cast<char*>(g.dev) + kFlagsBytes);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (which == 0)
    skinny_ar_diag_write<<<(int)blocks, kThreads, 0, stream>>>(reinterpret_cast<int4*>(x.data_ptr()), n8, slot);
  else
    skinny_ar_diag_read<<<(int)blocks, kThreads, 0, stream>>>(reinterpret_cast<int4*>(x.data_ptr()), n8, slot);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void skinny_ar_init(const std::string& path, int64_t rank, int64_t world, int64_t max_bytes) {
  TORCH_CHECK(!g.ready, "skinny_ar already initialised");
  TORCH_CHECK(world == 2 && (rank == 0 || rank == 1), "skinny_ar supports exactly 2 ranks");
  TORCH_CHECK(max_bytes % 16 == 0 && max_bytes > 0, "max_bytes must be a multiple of 16");
  g.slot_bytes = (size_t)max_bytes;
  g.map_bytes = kFlagsBytes + 4 * g.slot_bytes;
  // page-align
  const size_t page = (size_t)sysconf(_SC_PAGESIZE);
  g.map_bytes = (g.map_bytes + page - 1) / page * page;
  int fd = open(path.c_str(), O_RDWR | O_CREAT, 0600);
  TORCH_CHECK(fd >= 0, "skinny_ar: cannot open ", path, ": ", strerror(errno));
  TORCH_CHECK(ftruncate(fd, (off_t)g.map_bytes) == 0, "skinny_ar: ftruncate failed");
  g.host = mmap(nullptr, g.map_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  close(fd);
  TORCH_CHECK(g.host != MAP_FAILED, "skinny_ar: mmap failed: ", strerror(errno));
  // Zero MY flags and data slots so a stale file from an earlier run cannot
  // satisfy a wait. Caller must barrier after init on both ranks.
  unsigned* flags = reinterpret_cast<unsigned*>(g.host);
  for (int par = 0; par < 2; par++) {
    memset(flags + ((size_t)(par * 2 + rank) * kMaxChunks) * kFlagStride, 0,
           (size_t)kMaxChunks * kFlagStride * sizeof(unsigned));
    memset(reinterpret_cast<char*>(g.host) + kFlagsBytes + (size_t)(par * 2 + rank) * g.slot_bytes,
           0, g.slot_bytes);
  }
  C10_CUDA_CHECK(cudaHostRegister(g.host, g.map_bytes, cudaHostRegisterMapped));
  C10_CUDA_CHECK(cudaHostGetDevicePointer(&g.dev, g.host, 0));
  C10_CUDA_CHECK(cudaMalloc(&g.seq_dev, 2 * sizeof(unsigned)));
  C10_CUDA_CHECK(cudaMemset(g.seq_dev, 0, 2 * sizeof(unsigned)));
  g.done_dev = g.seq_dev + 1;
  g.rank = (int)rank;
  g.world = (int)world;
  g.path = path;
  g.ready = true;
}

int64_t skinny_ar_max_bytes() { return g.ready ? (int64_t)g.slot_bytes : 0; }

bool skinny_ar_ready() { return g.ready; }

// In-place fp16 sum across the two ranks. Requires contiguous, 16 B aligned,
// numel*2 <= max_bytes.
torch::Tensor skinny_ar_all_reduce(torch::Tensor x, int64_t mode) {
  TORCH_CHECK(g.ready, "skinny_ar not initialised");
  TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kFloat16 && x.is_contiguous(), "fp16 contiguous");
  const int64_t bytes = x.numel() * 2;
  TORCH_CHECK(bytes % 16 == 0 && bytes <= (int64_t)g.slot_bytes, "size/alignment");
  TORCH_CHECK((reinterpret_cast<uintptr_t>(x.data_ptr()) & 15) == 0, "16 B alignment");
  const c10::cuda::OptionalCUDAGuard guard(x.device());
  auto out = torch::empty_like(x);
  const int n8 = (int)(bytes / 16);
  int blocks = (n8 + kThreads - 1) / kThreads;
  if (blocks > kMaxBlocks) blocks = kMaxBlocks;
  if (blocks < 1) blocks = 1;
  unsigned* flags = reinterpret_cast<unsigned*>(g.dev);
  int4* data = reinterpret_cast<int4*>(reinterpret_cast<char*>(g.dev) + kFlagsBytes);
  auto stream = at::cuda::getCurrentCUDAStream();
  const int4* xp = reinterpret_cast<const int4*>(x.data_ptr());
  int4* op = reinterpret_cast<int4*>(out.data_ptr());
  const int slot_int4 = (int)(g.slot_bytes / 16);
  if (mode == 0)
    skinny_ar2_kernel<0><<<blocks, kThreads, 0, stream>>>(xp, op, n8, flags, data, g.rank, slot_int4, g.seq_dev, g.done_dev);
  else if (mode == 2)
    skinny_ar2_kernel<2><<<blocks, kThreads, 0, stream>>>(xp, op, n8, flags, data, g.rank, slot_int4, g.seq_dev, g.done_dev);
  else
    skinny_ar2_kernel<1><<<blocks, kThreads, 0, stream>>>(xp, op, n8, flags, data, g.rank, slot_int4, g.seq_dev, g.done_dev);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}


// Fused all-reduce + residual + Gemma RMSNorm. Returns (normalized fp16, residual fp32).
std::vector<torch::Tensor> skinny_ar_gemma_norm(torch::Tensor x, torch::Tensor residual,
                                                torch::Tensor weight, double eps) {
  TORCH_CHECK(g.ready, "skinny_ar not initialised");
  TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kFloat16 && x.is_contiguous() && x.dim() == 2);
  const int M = (int)x.size(0), H = (int)x.size(1);
  TORCH_CHECK(residual.is_contiguous() && residual.sizes() == x.sizes(), "residual shape");
  TORCH_CHECK(weight.is_contiguous() && weight.numel() == H, "weight");
  TORCH_CHECK(H % 8 == 0 && H <= kNormThreads * 32 && M >= 1 && M <= kMaxChunks, "H/M");
  TORCH_CHECK((int64_t)M * H * 2 <= (int64_t)g.slot_bytes, "message too large");
  TORCH_CHECK((reinterpret_cast<uintptr_t>(x.data_ptr()) & 15) == 0, "alignment");
  const c10::cuda::OptionalCUDAGuard guard(x.device());
  auto out = torch::empty_like(x);
  auto res_out = torch::empty({M, H}, x.options().dtype(torch::kFloat32));
  unsigned* flags = reinterpret_cast<unsigned*>(g.dev);
  int4* data = reinterpret_cast<int4*>(reinterpret_cast<char*>(g.dev) + kFlagsBytes);
  auto stream = at::cuda::getCurrentCUDAStream();
  const int4* xp = reinterpret_cast<const int4*>(x.data_ptr());
  half* op = reinterpret_cast<half*>(out.data_ptr<at::Half>());
  float* rp = res_out.data_ptr<float>();
  const int slot_int4 = (int)(g.slot_bytes / 16);
#define LAUNCH_NORM(TRes, TW, rptr, wptr)                                                   \
  skinny_ar2_gemma_norm_kernel<TRes, TW><<<M, kNormThreads, 0, stream>>>(                    \
      xp, rptr, wptr, op, rp, H, (float)eps, flags, data, g.rank, slot_int4, g.seq_dev, g.done_dev)
  if (residual.dtype() == torch::kFloat32 && weight.dtype() == torch::kFloat32)
    LAUNCH_NORM(float, float, residual.data_ptr<float>(), weight.data_ptr<float>());
  else if (residual.dtype() == torch::kFloat32 && weight.dtype() == torch::kFloat16)
    LAUNCH_NORM(float, half, residual.data_ptr<float>(), reinterpret_cast<const half*>(weight.data_ptr<at::Half>()));
  else if (residual.dtype() == torch::kFloat16 && weight.dtype() == torch::kFloat32)
    LAUNCH_NORM(half, float, reinterpret_cast<const half*>(residual.data_ptr<at::Half>()), weight.data_ptr<float>());
  else if (residual.dtype() == torch::kFloat16 && weight.dtype() == torch::kFloat16)
    LAUNCH_NORM(half, half, reinterpret_cast<const half*>(residual.data_ptr<at::Half>()), reinterpret_cast<const half*>(weight.data_ptr<at::Half>()));
  else
    TORCH_CHECK(false, "residual/weight must be fp16 or fp32");
#undef LAUNCH_NORM
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, res_out};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("init", &skinny_ar_init, "init(shm_path, rank, world, max_bytes)");
  m.def("all_reduce", &skinny_ar_all_reduce, "out-of-place 2-rank fp16 all-reduce via host memory (x, mode)",
        py::arg("x"), py::arg("mode") = 1);
  m.def("max_bytes", &skinny_ar_max_bytes, "max message bytes");
  m.def("all_reduce_gemma_norm", &skinny_ar_gemma_norm,
        "fused 2-rank all-reduce + residual + Gemma RMSNorm -> (fp16 normalized, fp32 residual)");
  m.def("ready", &skinny_ar_ready, "initialised?");
  m.def("diag", &skinny_ar_diag, "phase diagnostics (x, which, blocks)");
}
