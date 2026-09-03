# 08 DeepEP 与 MoE 通信算子

> 来源：`手记.md` 7.13、7.22、7.27、7.29–7.31、8.4、8.6、8.11、8.18、8.24–8.26、8.28
> 参考：工作区 `DeepEP/`（源码）、`IBGDA详解.md`、`combine优化plan.md`、`GLM5.1-S5000调优记录.md`
> 关联：[06 分布式并行](./06_分布式并行.md)、[07 互连与网络技术](./07_互连与网络技术.md)、[09 调优案例复盘](./09_调优案例复盘.md)

---

## 1. DeepEP 是什么

**DeepEP** 是面向 MoE / 专家并行（EP）的通信库，提供高吞吐、低延迟的 **all-to-all GPU kernel**（即 MoE 的 **dispatch** 与 **combine**），支持 FP8 等低精度。两条路径：
- **normal kernel**（NVLink + RDMA 转发）：高吞吐，适合训练/推理 prefill，支持控制 SM 数。
- **low-latency kernel**（纯 RDMA）：低延迟，适合推理 decode，支持 hook 式通信-计算重叠、**不占 SM**。

---

## 2. 核心数据结构与概念

### 2.1 Buffer / PE / rdma_buffer

- **Buffer**：每个 rank 一个 `deep_ep.Buffer()`，一个通信 stream（构造器内 `comm_stream`）。
- **PE（Processing Element）**：DeepEP 设计中**每张 GPU 是一个独立的 NVSHMEM PE**，每张 GPU 的 VRAM 里各有一份 `rdma_buffer`。远端 GPU 发起 IBGDA put 时，目标地址是某张特定 GPU 的对称堆偏移量。**8 个独立的 rdma_buffer，有并行与带宽优势**（8.11）。
- **rdma_buffer**：跨节点 RDMA 的暂存区（对称堆）。

### 2.2 channel（逻辑通信通道）

**一个 sender/receiver SM 对组成一个 channel**（8.4）：
- **sender 负责 tail**：每写入一个 token 到 buffer，`channel_tail_idx` 递增；
- **receiver 负责 head**：每读出（消费）一个 token，`channel_head_idx` 递增；
- sender 写完后 `st_release_sys_global(channel_tail_idx, ...)` **发布 tail 指针**；
- receiver `ld_acquire_sys_global(channel_tail_idx)` 拿最新 tail，判断 `tail - head > 0` 决定有多少新数据可读。

### 2.3 buffer / L2 / DRAM 的关系（❓答疑）

> 记录 8.4。

**💡 解答**：**buffer 是逻辑概念**，是 DRAM 里划分出的一块**环形缓冲区**，用于跨 GPU 通信。DRAM 是物理显存；L2 是 GPU 片上共享缓存。跨 GPU 通信的数据最终落在对端 GPU 的 DRAM（buffer 所在），读写过程可能经过 L2（是否经过/是否留脏副本，取决于用哪种 load/store 原语，见 §5）。

### 2.4 kNumRanks / send buffer / recv buffer

- **kNumRanks**：通信组里 GPU 数量，决定每个 token 最多被几个 GPU 的数据参与 reduce（8.4）。
- **send/recv buffer 槽号互补**（7.31）：同一次 put 的 src/dst 偏移必须相同，而语义上「我发往 X」与「X 收自我」是同一份数据——所以 **send 用目标编号，recv 用来源编号**。
  - 例：rank0（节点A，rdma_rank=0）发给节点B（rdma_rank=1）时，`recv_buffer(0)`（本地 recv 段第 0 槽 = 目标节点「来自 A」的接收槽）+ `send_buffer(1)`（本地 send 段第 1 槽 = 「发往 B」的暂存区），`pe=1`。
  - 一句话：**节点 A 把「发给节点 B 的暂存槽」内容，写到节点 B 的「来自节点 A 的接收槽」**。

---

## 3. 普通 internode vs 低延迟 internode（IBRC vs IBGDA）

> 记录 7.30 的完整对比表 + 后续补充。

| 维度 | 传统 IBRC | IBGDA |
|---|---|---|
| 谁写 Doorbell | CPU | GPU 线程 |
| 谁组装 WR | CPU 驱动 | GPU 线程 |
| 谁轮询 CQ | CPU | GPU 线程 |
| 是否占 communication SMs | 是，需要单独通信 kernel | 否，inline 在计算 kernel 里 |
| CPU 数据路径 | 每次通信都参与 | 初始化后不再参与 |
| 延迟 | 高（CPU 同步 + kernel 启动） | 低（GPU 线程直接触发） |
| 适用场景 | 高吞吐大消息（DeepEP normal） | 低延迟小消息（DeepEP low-latency） |

**普通 RDMA dispatch kernel 占着 SM，主要在做**：数据打包、NVLink 转发、RDMA 批量协调、接收解包，以及各种 barrier / head-tail / FIFO 同步。

### 3.1 低延迟模式为什么可以不需要通信 SM（❓答疑）

> 记录 7.30。

**💡 解答**：把「数据整理」和「RDMA 发送（`nvshmemi_ibgda_put_nbi_warp`）」**熔进了同一个小 kernel**，并支持 **hook 回调**——通信动作 inline 在计算 kernel 里，不需要单独的通信 kernel 占用额外 SM。

### 3.2 LowLatencyLayout 双缓冲

用 `LowLatencyLayout` 把 `rdma_buffer_ptr` 切成两块（`buffers[0]/buffers[1]`，`deep_ep.cpp:1036-1037`），**双缓冲交替使用**（`low_latency_buffer_idx ^= 1`）。好处：省 cleanup kernel + CPU 侧流水 enqueue。

### 3.3 "NVLink domain → RDMA domain" 转发（❓答疑）

> 记录 7.22。

**💡 解答**：先发 RDMA 到对面节点的 rdma buffer，再通过 **NVLink/IPC 转发到目标 nvl_rank**。即跨节点数据先落到对面节点的「入口 rdma buffer」，再由节点内 NVLink 域转发给真正要收的那张卡。

---

## 4. dispatch / combine 的流程

### 4.1 notify_dispatch vs cached_notify（8.28）

**notify_dispatch**（完整版「通知」内核，`internode.cu:284`）：dispatch 第一步**不是搬数据，而是先让所有 rank 互相同步元数据**：
1. 排空上一轮在途 IBGDA WQE，做 NVL/跨节点 barrier，清零本轮要复用的 RDMA/NVL 交换区；
2. **交换 token 计数**（我要发给每个 rank/每个 expert 多少 token）广播给所有对端；
3. 计算布局表（`rdma_channel_prefix_matrix`、`gbl_channel_prefix_matrix` 等，即每个 channel 的 token 放哪、接收端每段从哪读），把收端计数写回 `moe_recv_counter`；
4. host 自旋等计数到齐，才能分配输出 tensor。
→ 之后才是真正搬数据的 dispatch kernel。

**cached_notify**（布局已缓存时的「瘦身版」，`internode.cu:3266`）：上面那套「计数交换 + 布局计算 + 全局 barrier + host 自旋等待」的结果**只取决于路由（topk_idx）**。推理场景路由往往逐轮不变（或用 `get_dispatch_layout` 预先算好），把布局表通过 handle 传回来，元数据计算全部跳过——**只剩每轮必须做的：quiet + barrier + 清零 head/tail/meta 标志区**（即注释 "Just a barrier and clean flags"）。

### 4.2 dispatch 的 send loop：消息提交与拷贝、all-to-all 重排（❓答疑）

> 记录 8.18 的问题，据 `internode_ll.cu` 结构回答。

**❓ 消息提交与拷贝是什么，用在什么阶段？**

**💡 解答**：dispatch 的 send 阶段，每个 token 要完成两件事：
1. **打包/拷贝**：把 hidden data + FP8 scales + 源索引（`src_info`）组成一条消息，写入发送暂存区（本地 rank 直写对端 recv slot，远程 rank 先拷到本地 send buffer）。
2. **消息提交**：写完数据后**发布 tail 指针**（本地 P2P 用 `st_release`），或**发出 IBGDA put**（写 WQE + doorbell）——这是「告诉接收方/网卡有数据可收」的动作。

源码对应：`internode_ll.cu` 里 "Message package: hidden data, FP8 scales, index at source" → "Issue IBGDA sends"（`Copy directly to local rank, or copy to buffer and issue RDMA`）→ "Receiving and packing"。

**❓ MoE all-to-all 需要做各种数据重排与发送，具体在哪里体现？ll 版为何"直接发送"？**

**💡 解答**：all-to-all 的本质是**每个 token 要路由到「目标 rank、目标 expert」的 slot**——这就是「重排」。
- **normal 版**：先跑布局 kernel（`layout.cu`）统计每个 rank/RDMA-rank/expert 收多少 token、`is_token_in_rank`，再按布局**分组打包**发送，host 还要等计数。
- **ll 版（低延迟）**：布局简单（decode 场景每 rank 最多 `num_max_dispatch_tokens_per_rank`），**直接把 token 写入目标 (rank, expert) 对应的 recv slot**——local 直接 SM 写、remote 拷到 send 区再 IBGDA put，**无需 host 等计数、无需单独 layout 阶段**，这就是「ll 版直接发送」。

### 4.3 combine 的 receiver（❓答疑）

**❓ combine kernel 的 receive 端是哪个 GPU？**

**💡 解答**：combine 的 receive 端就是**当前运行 combine kernel 的 GPU 本身**——它要读的数据是 **kNumRanks 个 GPU 发到它自己 buffer** 里的（每个专家结果按 token 归约回发出 token 的卡）。

### 4.4 internode.cu 的数据流（8.6）

```
远端节点 global memory  --[RDMA/IBGDA]-->  本卡 global memory(rdma_buffer)
本卡 global memory(rdma_buffer)  --[TMA]-->  本卡 shared memory(smem，本 SM 私有)
本卡 shared memory  --[TMA/store]-->  本卡 global memory(nvl_buffer，经 NVLink 给邻居卡)
```

---

## 5. intranode 0804 优化：显式缓存策略 + 内存级并行（核心案例）

### 5.1 现象与根因

自建 profiler 测出：combine 的 **receiver 端 reduce 循环占 ~75%**，其中**数据读取占 60-70%**，每读一个 token 约 **199 微秒**，且**与并发数无关** → 是**每次读取本身的延迟太高**，不是带宽不够。

**根因**：MUSA 的 `volatile_load`（所有 load 原语都映射到它）默认带 **CFI（Cache Flush and Invalidate）** 策略——**每次读取都把整个 GPU 共享的 L2 缓存（SLC）全部刷一遍再作废**。56 个 SM、上千 warp 同时做，等于所有人抢着刷同一块缓存，互相串行化。

### 5.2 六条策略

1. **数据读取改用 `__ldcg`（最大单项收益 +12%）**：走 L2 缓存路径读取，不触发 CFI 全局刷缓存。远端 GPU 写的数据本就在接收方 DRAM 里，经 L2 读即可。每 token 读取 ~199us → ~143us。
2. **发送端改用 `__stwt`（write-through 写，+14%）**：写数据时**直接写到接收方 DRAM，不在发送方 L2 留副本**。之前用普通写（`st_na_global`），数据先落发送方 L2 变脏行，接收方读时还要跨 PCIe 去发送方 L2 捞脏行。
3. **receiver 的 reduce 结果写改用 `__stcs`（streaming store，+7%）**：适合「写一次、读一次」，写时不往缓存填，减少缓存污染。
4. **4 路内存级并行（MLP）**：同时发多个内存读请求，在硬件流水线里并行执行，用总延迟除以并发数摊薄开销。改每次迭代发 4 组、每组 `kNumRanks` 个 load 背靠背发（**实现上通过 `#pragma unroll` 展开循环、同时发出多个位置的 Load**）。试过 8 路但寄存器溢出反而变慢，**4 路是平衡点**。
5. **无条件 load + 谓词化 add**：把 `topk_ranks`/`slot_indices` 预填安全默认值（rank 0、slot 0），**无条件对所有 rank 发 load**（多读安全地址无副作用），只在加法那步用 `if (j < num_topk_ranks)` 守卫。整个 load 循环可编译期完全展开、零分支，消除 warp 内 divergence。
6. **轮询类读取用 NO_CFI volatile**：head/tail 索引轮询需 volatile 语义（读到最新值），用 `volatile_load(ptr, 4)` 跳过 CFI，避免每次轮询刷全局缓存。

### 5.3 普通 store 的 dirty-L2 流程（为什么 write-through 更快）

```
1. sender 把数据存在 L2，标记 dirty，再发到 receiver DRAM
2. receiver L2 miss → 转发到内存控制器 → 硬件缓存一致性协议检查其他 GPU L2 是否有这行的脏副本
   - 若有：数据还没到 DRAM → 跨 PCIe 取远端 L2 脏行
3. sender 写完调用 __threadfence_system()：
   强制 sender L2 所有脏行写回 → 脏行经互连写到 receiver DRAM
   （缓存行带物理地址，sender L2 知道这行不属于自己）
   → sender L2 这些行 dirty→clean（或淘汰）→ receiver DRAM 里是最新数据
```

---

## 6. EP 下 dispatch 耗时的重叠方案

### 6.1 two-batch overlap

把一个 batch 分为两个 micro-batch，让计算和通信重叠进行：

```
B0: Attention -> Dispatch -----> MLP -> Combine
B1:                     Attention -> Dispatch -----> MLP -> Combine
```

### 6.2 single-batch overlap（SBO）

用 2 个流，把 combine 通信与 down-GEMM 计算并行：**不必等所有专家的 down-GEMM 全做完才开始 combine——做完一个专家就回收一个专家**；以及 dispatch/combine 与 shared experts 计算重叠。

> SBO 在 MUSA 版的卡点（combine 缺 `overlap/comp_signal/num_sms/block_m/threshold` 参数）见 [09](./09_调优案例复盘.md) 案例 3。

---

## 7. 源码结构与移植经验

### 7.1 `.py` 是干什么的 + 调用链（❓答疑）

> 记录 7.27 的「算子库里的 .py 是干什么的」。

**💡 解答**（8.24）：`.py` 是 **pybind11 绑定** 的 Python 封装层。**pybind11** 是轻量级 C++ 库，把 C++ 代码绑定到 Python。整个调用链：

```
Python API (buffer.py) → pybind11 → Buffer 方法 (host API: deep_ep.cpp) → host wrapper (*.cu 中) → device kernel
```

### 7.2 关键源文件结构（8.24）

- **`csrc/kernels/layout.cu`**：分发布局计算 GPU kernel。输入 top-k 索引，统计每个 rank/RDMA-rank/expert 收多少 token，标记每个 token 属于哪些 rank（`is_token_in_rank`）。用 SM 分块并行统计（前若干 SM 按 expert 分块、后若干 SM 按 rank 分块，线程内计数后 `__syncthreads` 归约）。是 MoE EP dispatch 的第一步（告诉 dispatch kernel 数据往哪发）。
- **`csrc/config.hpp`**（header-only）：缓冲区大小估算与低延迟布局。`Config` 结构体存 `num_sms` 与 NVL/RDMA chunked send/recv token 上限，构造时对齐断言校验；`get_nvl_buffer_size_hint`/`get_rdma_buffer_size_hint` 按 channel×rank×token×hidden 估算最小显存；`LowLatencyBuffer/LowLatencyLayout` 把 RDMA buffer 切成两组对称子缓冲区供低延迟路径用。
- **`csrc/kernels/runtime.cu`**：NVSHMEM 运行时生命周期 + 节点内 barrier kernel（`intranode::barrier`、`internode::get_unique_id/init/alloc/free/barrier/finalize/selfcheck`）。

### 7.3 移植优化经验（8.25）

1. **CUDA 哪些东西 MUSA 做不了 → 绕开**；
2. **MUSA 自己的功能怎么用 → 从 SDK/官方代码里找例子**；
3. **profiling 找移植版与系统版区别**。

---

## 8. profiling trace 中的 internode 算子是哪来的（❓答疑）

> 记录 7.22。

**💡 解答**：来自 **DeepEP 的跨节点路径**（`internode.cu` / `internode_ll.cu` 里的 RDMA dispatch/combine kernel）。当时的移植版是「**卡间 intranode（MUSA）+ cuda internode（跨节点 RDMA）**」（`mt-deepep`），trace 里出现的 internode 算子就是这些跨节点 RDMA 通信 kernel（IBGDA put 等）。
