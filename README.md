## 主题导览

| 文件 | 内容 |
|---|---|
| [01 部署与环境搭建](./01_部署与环境搭建.md) | 裸机/Docker、conda、CUDA/驱动、RAID0、OOM、MUSA SDK、`.cu/.mu` 工作流 |
| [02 推理框架与引擎](./02_推理框架与引擎.md) | vLLM、SGLang、TensorRT、ONNX、TVM、CUDA graph、chunked prefill、RadixAttention |
| [03 模型与算法](./03_模型与算法.md) | DS V4 FP4 MoE/MTP、GLM5.1 PD/NSA、投机解码谱系、LoRA、Mooncake |
| [04 性能分析方法论](./04_性能分析方法论.md) | 指标、profiling 工具、compute/memory-bound、并发扫描、屋顶模型、wall clock |
| [05 GPU 硬件与编程基础](./05_GPU硬件与编程基础.md) | 执行层级、内存层级、SIMT、stream、TMA、DMA、MPS/MIG |
| [06 分布式并行](./06_分布式并行.md) | DP/SP/CP/TP/PP、EP、scale-up/out、集合通信、tp/ep/dp 公式 |
| [07 互连与网络技术](./07_互连与网络技术.md) | PCIe/NVLink、RDMA/IBGDA/RoCEv2/DCQCN、NVSHMEM、ACE/TME/TMA、MUSA vs CUDA |
| [08 DeepEP 与 MoE 通信算子](./08_DeepEP与MoE通信算子.md) | DeepEP 架构、normal vs low-latency、notify、dispatch/combine、0804 缓存优化、源码结构 |
| [09 调优案例复盘](./09_调优案例复盘.md) | H20 DS V4、8×H20 all-reduce skew、S5000 GLM5.1、mpirun EAGAIN |

---
