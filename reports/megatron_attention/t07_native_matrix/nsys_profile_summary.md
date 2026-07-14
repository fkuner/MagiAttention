# T07 Native Nsight Systems Profiles

> Diagnostic captures only. Profiled timings are perturbed and are not headline results.

- Status: `pass`

| Case | Status | Expected ranges | Forward | Backward | Profiled median (ms) | CV |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `t07_probe_dense_sbhd_cp4_all_gather` | `pass` | 12 | 12 | 12 | 7.363002 | 1.239988 |
| `t07_probe_dense_thd_cp4_p2p` | `pass` | 12 | 12 | 12 | 21.611413 | 0.186516 |
| `t07_probe_dsa_sbhd_cp4_allgather` | `pass` | 12 | 12 | 12 | 24.680485 | 0.267144 |

## t07_probe_dense_sbhd_cp4_all_gather top kernels

| Time | Instances | Median ns | Max ns | Kernel |
| ---: | ---: | ---: | ---: | --- |
| 89.2% | 48 | 68975.5 | 145240292 | `ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` |
| 5.1% | 72 | 29007.5 | 1276284 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` |
| 3.2% | 24 | 68192.0 | 841662 | `ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` |
| 0.2% | 72 | 5696.0 | 6016 | `void at::native::<unnamed>::indexSelectSmallIndex<c10::BFloat16, int, unsigned int, (int)2, (int)2, (int)-2>(at::cuda::detail::TensorInfo<T1, T3>, at::cuda::detail::TensorInfo<const T1, T3>, at::cuda::detail::TensorInfo<const T2, T3>, int, int, T3, long)` |
| 0.2% | 24 | 16016.0 | 26784 | `cudnn_generated_fort_native_sdpa_sm80_flash_fprop_wmma_f16_knob_3_64x64x256_4x1x1_kernel0_0` |

## t07_probe_dense_thd_cp4_p2p top kernels

| Time | Instances | Median ns | Max ns | Kernel |
| ---: | ---: | ---: | ---: | --- |
| 71.3% | 168 | 12128.0 | 7937431 | `ncclDevKernel_SendRecv(ncclDevKernelArgsStorage<(unsigned long)4096>)` |
| 16.0% | 72 | 33311.0 | 2043802 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` |
| 1.0% | 450 | 1760.0 | 2048 | `void at::native::vectorized_elementwise_kernel<(int)8, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char *, (unsigned long)3>>(int, T2, T3)` |
| 0.9% | 276 | 2528.0 | 3296 | `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambda() (instance 12)]::operator ()() const::[lambda(c10::BFloat16) (instance 1)]>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)` |
| 0.9% | 288 | 2336.0 | 2880 | `void at::native::<unnamed>::CatArrayBatchedCopy<at::native::<unnamed>::OpaqueType<(unsigned int)2>, unsigned int, (int)4, (int)64, (int)64>(T1 *, at::native::<unnamed>::CatArrInputTensorMetadata<T1, T2, T4, T5>, at::native::<unnamed>::TensorSizeStride<T2, (unsigned int)4>, int, T2)` |

## t07_probe_dsa_sbhd_cp4_allgather top kernels

| Time | Instances | Median ns | Max ns | Kernel |
| ---: | ---: | ---: | ---: | --- |
| 51.7% | 24 | 853179.5 | 10886232 | `ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` |
| 20.6% | 156 | 30128.0 | 1322171 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` |
| 7.3% | 36 | 127455.5 | 386623 | `ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` |
| 0.9% | 204 | 3296.0 | 4064 | `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambda() (instance 7)]::operator ()() const::[lambda(float) (instance 1)], std::array<char *, (unsigned long)2>, (int)4, TrivialOffsetCalculator<(int)1, unsigned int>, TrivialOffsetCalculator<(int)1, unsigned int>, at::native::memory::LoadWithCast<(int)1>, at::native::memory::StoreWithCast<(int)1>>(int, T1, T2, T4, T5, T6, T7)` |
| 0.8% | 144 | 3712.0 | 10208 | `void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, float>::operator ()(at::TensorIterator &)::[lambda(float, float) (instance 1)]>, unsigned int, float, (int)4, (int)4>>(T3)` |
