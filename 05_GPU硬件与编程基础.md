# 05 GPU 硬件与编程基础

> 来源：`手记.md` 6.24、7.2、7.9–7.10、7.21、8.3、8.6、8.14、8.18
> 关联：[07 互连与网络技术](./07_互连与网络技术.md)、[08 DeepEP与MoE通信算子](./08_DeepEP与MoE通信算子.md)

---

## 1. 执行层级（线程组织）

| 层级 | 含义 |
|---|---|
| **Grid** | 一次 kernel launch 的所有 blocks |
| **cluster** | 让多个 block 同时在多个 SM 上更高效地共享与协作 |
| **Block（线程块）** | GPU 调度基本单位，分配到 1 个 SM，其所有 threads 在该 SM 上并行执行同一 kernel 函数 |
| **warp（32 threads）** | SM 实际调度和执行的基本单位 |
| **Thread** | 最小执行单元 |

- **CTA（Cooperative Thread Array）**：CUDA 里基本可理解为「一个 thread block（线程块）」。
- `ceil` 后的 block 内含 warp 数 = block 的 warp 数。
- **Shared memory**：物理上在 SM 上，只在**同一个 thread block 内部线程**之间共享。
- **tensor core**：SM 中做 MMA（矩阵乘累加）的计算单元。
- **global memory**：整个 GPU 的所有 SM 之间共享。

### SIMT（❓+ 概念）

**❓ 如何理解 SIMT？**

**💡 解答**：写的是**标量线程代码**，一个 warp 中的 32 个线程**通常会同时执行同一条指令，但作用在不同的数据上**。同一 warp 里的线程理论上以 SIMT 方式一起执行同一条指令。

### occupancy / Waves Per MP / SM 驻留密度

- **MP（Multi-Processor）**：GPU 的计算单元（MUSA 对 SM 的称呼），可以同时驻留（resident）多个 warp 交替执行。
- **occupancy**：GPU 一个 SM 上「同时能跑多少 warp / thread block」的比例。
- **SM 驻留密度**：一个 SM 上同时放的 block 多少。
- **Waves Per MP** = 实际驻留的 warp 数 / MP 硬件支持的最大 warp 数（8.3）。

### 寄存器与驻留

- **regs/thread**：GPU 上每个线程的局部变量**不放在栈上，而是直接映射到物理寄存器**。编译器编译 kernel 时分析变量生命周期，为每线程分配固定数量寄存器。
- 线程占用寄存器过多 → 每 MP 可同时驻留的 warp 就少。因为每 MP 寄存器文件总量固定：

```
max_blocks_per_MP = 寄存器文件总量 / (regs_per_thread × threads_per_block)
```

---

## 2. 内存层级

### 2.1 硬件物理层级

```
CPU Host Memory
    ↓ PCIe / NVLink
GPU Device Memory：HBM / GDDR 显存
    ↓
L2 Cache：整个 GPU 跨 SM 共享
    ↓
每个 SM 内部：
    L1 Cache / Texture Cache / Shared Memory
    ↓
Register File：每个线程使用
```

### 2.2 CUDA 编程模型里的 memory spaces

| CUDA 名称              | 位置/本质                                       | 谁能访问                     | 典型用途                            |
| ---------------------- | ----------------------------------------------- | ---------------------------- | ----------------------------------- |
| `global memory`        | 主要在显存/HBM/GDDR 上，经过 L2/L1 cache        | 所有线程                     | 大 tensor、矩阵、模型参数、输入输出 |
| `local memory`         | 名字叫 local，但物理上通常也在 global memory 中 | 单个线程私有                 | register spill、大数组放不进寄存器  |
| `shared memory`        | SM 片上存储                                     | 一个 thread block 内线程共享 | tile 缓存、block 内协作             |
| `register`             | SM 内寄存器文件，最快                           | 单个线程私有                 | 临时变量、accumulator               |
| `constant memory`      | 物理上在 device memory，有 constant cache       | 所有线程只读                 | 小规模只读参数，广播访问            |
| `texture memory`       | 物理上在 device memory，通过 texture cache 访问 | 所有线程只读/特定访问模式    | 图像、空间局部性访问                |
| `surface memory`       | device memory 的一种访问路径                    | 所有线程                     | 图像/二维写入等                     |
| `TMEM / Tensor Memory` | Blackwell 新增片上存储                          | Tensor Core 相关指令使用     | MMA accumulator、中间 tile          |

### 2.3 统一内存与显式内存管理

- **unified memory**：`cudaMallocManaged` 分配、`cudaFree` 释放，**CPU、GPU 均可访问**。
- **显式内存管理**：`cudaMallocHost`（pinned host）、`cudaMalloc`（device）、`cudaMemcpy`（拷贝）。

---

## 3. 编程模型

### 3.1 内置变量

`threadIdx`、`blockDim`、`blockIdx`、`gridDim`，每个都是一个 **3 分量向量**（`.x`/`.y`/`.z`）。启动配置未指定的维度默认为 1。例：`threadIdx.x` 取 0 到 `blockDim.x-1`。

### 3.2 tile 编程

程序员**逻辑上对 block 操作，数据上对 tile 操作**。

### 3.3 同步

`__syncthreads()` 仅负责同步**同一个线程块内**的各个线程（不能跨块同步）。

### 3.4 cuda context

GPU 上运行 CUDA 程序所需的「运行环境」，在**第一次调用需要 GPU Context 的 Runtime API 时创建**，被所有 Host 线程共享。

### 3.5 lane

32 个线程在 warp 内的编号：`lane_id = threadIdx.x % 32`。

### 3.6 stream（❓答疑）

> 记录 6.26/6.29 反复出现的问题。

**❓ stream 是什么？为什么几个 GPU 可以共用一个 stream？一个 stream 可以多个 GPU 并行吗？一个 CUDA stream 不能跨 GPU，为什么不同 GPU 上都有 stream23？**

**💡 解答**：
- **stream（流）**：**单个 device 上的一个「有序执行队列」**——同一条 stream 里的 kernel/操作按提交顺序串行执行；不同 stream 之间可以并行（用于重叠计算与通信/拷贝）。
- **一个 CUDA stream 不能跨多个 GPU 执行 kernel**：stream 是 **per-device** 的对象，隶属于某一张卡，跨卡必须靠「多卡通信」而不是「共用一个 stream」。
- **为什么不同 GPU 上都有 stream23？**：每个 device 有自己**独立的 stream 编号空间**。「stream23」只是各卡自己那套 stream 里的第 23 号，它们是**不同的对象**（各自在自己的卡上、彼此独立、可各自并行），只是编号恰好相同。GUI 里看到「多张卡都有 stream23」不代表共享同一条流。

---

## 4. kernel launch 与执行（❓答疑）

**❓ kernel launch 从 CPU 到 GPU 的过程？**

**💡 解答**（记录 6.24 + 7.21）：完整链路是「编译 → 前端/后端 → host 端代码 → 运行时」，执行上有三个层次：

```
host 调用 musaLaunchKernel
        ↓
  host 端命令缓冲区（driver 内部）
        ↓  ← 这一步需要显式或隐式 flush
  GPU 硬件队列（真正排队等执行）
        ↓
  GPU 实际执行
```

- 关键点：**launch（host 把命令塞进队列）≠ execute（GPU 真正跑）**。host 先跑一大截把命令都塞进命令缓冲区，GPU 何时跑取决于流的调度与 flush。这正是 [09](./09_调优案例复盘.md) 里「单流提交饥饿」空泡的机制基础。

---

## 5. 硬件特性与高级机制

### 5.1 Hopper GEMM pipeline（简化）

| 阶段 | 组件 | 作用 |
|---|---|---|
| 1 | **TMA producer** | 把 A_tile/B_tile 从 global memory 搬到 shared memory |
| 2 | **mbarrier** | 标记 shared memory tile 是否 ready |
| 3 | **WGMMA consumers** | 从 shared memory 读 A/B tile，执行 Tensor Core GEMM |
| 4 | **Epilogue** | 对结果做 scale、bias、activation，写回 global memory |

### 5.2 TMA（Tensor Memory Accelerator）

Hopper 架构硬件异步批量拷贝单元：单条指令 `cp.async.bulk` 就能把一大块数据（几百字节到几十 KB）从 global memory **异步**搬进 shared memory，搬完自动通知 mbarrier。**整个 warp 不用参与搬**，线程该干啥干啥。

> MUSA 的对标物是 **TME（Tensor Memory Engine）**，见 [07 互连与网络技术](./07_互连与网络技术.md)。

### 5.3 DMA / Copy Engine

- **DMA**：芯片上的一个**独立硬件拷贝引擎**，不占用 SM 的 store 带宽，有自己的指令集（如 LDMA）。
- NVIDIA GPU 内部有专门 **Copy Engine**，负责：
  - **H2D**：Host（CPU 内存）→ Device（GPU 显存）
  - **D2H**：Device → Host
  - **D2D**：Device → Device（显存内拷贝）
- 这些传输**完全不需要 SM 参与**，是硬件异步完成的——这就是 DMA 的本质。

### 5.4 MPS / MIG

- **NVIDIA MPS**：软件层面，多进程共享一张 GPU。
- **MIG**：硬件层面，把一张 GPU 切成多个硬件隔离实例，明确切分 SM、显存等资源。

### 5.5 warp specialization

在一个 block 内让不同 warp 分工，目标是**计算与访存重叠，隐藏访存延迟**（7.21）。

---

## 6. MoE 相关编程要点（衔接）

- MoE 在 **Gate-Up GEMM 激活之后 intermediate size 很大**，往往需要**量化后再做 Down GEMM**（7.21）——这是 MoE 专家层量化 GEMM 成为最大开销的背景（见 [08](./08_DeepEP与MoE通信算子.md) 与 [09](./09_调优案例复盘.md)）。
