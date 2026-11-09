# 12 GPU 软件栈与算子执行

> 来源：`Diary/2026-10-10.md`、`Diary/2026-10-11.md`、`Diary/2026-10-15.md`
> 参考：工作区 `landscape.md`（§二那张 ASCII 图的展开版）、本机 PyTorch 头文件 `torch/include/ATen/core/dispatch/Dispatcher.h`、`c10/core/DispatchKey.h`
> 关联：[05 GPU 硬件与编程基础](./05_GPU硬件与编程基础.md)（§4 kernel launch）、[04 性能分析方法论](./04_性能分析方法论.md)（profiling / 插桩）

---

## 1. 为什么要拆清「软件栈」

做 profiling、插桩或故障定位时，「驱动」「运行时」「算子」**都不是单层**。一次 `torch.mm(a, b)` 从 Python 到 GPU，要依次穿过：

- **框架层**：Python → `libtorch` → **ATen 算子 dispatch**（把 `aten::mm` 路由到具体实现）；
- **运行时层**：`libcudart.so`（`cudaLaunchKernel`）；
- **用户态驱动**：`libcuda.so`（`cuLaunchKernel`、PTX→SASS JIT）；
- **内核态驱动**：`nvidia.ko` / `nvidia-uvm.ko`（显存、页表、命令缓冲、runlist）；
- **设备侧**：GSP 固件 + GPU 硬件（GigaThread → SM → warp → SASS）。

分不清卡在哪条缝上，就不知道该把探针插在哪里。下面先给分层与可观测点，再自上而下走一遍。

### 1.1 分层与可观测点

| 层 | 位置 | 记录单位 | 常用探针 |
|---|---|---|---|
| **L0** | Python / `libtorch` / ATen dispatcher | **ATen 算子**（`aten::conv2d`） | torch profiler、`TorchDispatchMode` |
| **L1** | `libcudart` / `libcuda` 宿主库边界；`ioctl`/`mmap` 的 OS 边界 | **CUDA API 调用**（`cuLaunchKernelEx`） | uprobe、`LD_PRELOAD` |
| **L2** | `nvidia.ko` / `nvidia-uvm.ko` 内部 | 驱动内部函数/事件 | `struct_ops` 等内核挂点 |
| **L3** | kernel 内 SASS 指令流 | 指令/函数 | kernel 内插桩 |
| **L4** | L1/L2 cache、HBM、ECC、PMU | 硬件事件 | 硬件计数器（只读） |

### 1.2 全景走位图

```text
Python:  torch.mm(a, b)
   ↓
C++:     at::_ops::mm::call(a, b)                    [ATen/ops/mm_ops.h:26]
   ↓
Dispatcher::call(op, a, b)                           [Dispatcher.h:785]
   ├─ ① getDispatchKeySetUnboxed(a, b)   ← 从参数算 key（CUDA + Autograd 都置位）
   ├─ ② op.lookup(dispatchKeySet)        ← 查表拿到实现        [Dispatcher.h:798]
   ├─ ③ if (profiler 开着) → SlowPath    ← ★ RECORD_FUNCTION 就在这 [Dispatcher.h:802]
   └─ ④ kernel.call(op, keyset, a, b)    ← 真正跳进去
         ↓
      Autograd 版（记录计算图）
         ↓  redispatch（★ 不重复记 RecordFunction，Dispatcher.h:848）
      CUDA 版 → cuBLAS → 具体的 kernel 实现
   ↓
libcudart.so   cudaLaunchKernel()
   ↓
libcuda.so     cuLaunchKernel() / PTX JIT  ← L1 宿主库边界（uprobe 热点）
   │                                            ★ 驱动在这里有「用户态那一半」
═══ ioctl / mmap  on /dev/nvidia* ═══        ← ★ 用户/内核的缝
   │                                            （L1 的「OS 边界」就指这条缝）
nvidia.ko   分配显存 / 建页表 / 组装命令缓冲 / 写 GPU doorbell(MMIO)
nvidia-uvm  页错误 / 预取 / 驱逐 / 迁移     ← L2 驱动内 hook（struct_ops 挂这里）
   │                                            ★ 驱动在这里有「内核态那一半」
═══ PCIe / NVLink / 命令队列 / runlist ═══    ← ★ 宿主/设备的物理缝
   │
GPU：GigaThread 引擎取 runlist → 分 block 到 SM → 硬件 warp 调度器
   │
执行 SASS 指令流                             ← L3 kernel 内插桩
   │
L1/L2 cache、HBM、ECC 逻辑、PMU 计数器       ← L4 硬件计数器（只读）
```

**结论**：**驱动**横跨最上面那条断点（用户态 `libcuda.so`）和最下面那条（内核态 `nvidia.ko`），并且有一小块职责被卸载到设备侧的 GSP 固件里；**算子 dispatch** 则在驱动之上多出一层「语义层」，决定 `aten::mm` 到底落到哪一份实现。

---

## 2. 框架层：ATen 算子与 dispatch

### 2.1 先分清两个词

| 词 | 含义 | 例子 |
|---|---|---|
| **算子**（operator / ATen op） | 一个**语义单位**——「做矩阵乘」这件事，**不是一段代码** | `aten::mm`、`aten::add`、`aten::conv2d`、`aten::layer_norm` |
| **dispatch** | **派发 / 路由**——把这一个语义单位，路由到当前环境下**正确的那一份实现** | `aten::mm` 在 CUDA 上 → cuBLAS；在 CPU 上 → `mm_cpu`；稀疏张量上 → sparse mm |

**关键点在「实现是复数」**：`aten::mm` 这一个名字底下，挂着几十份不同的实现。

### 2.2 为什么需要 dispatch —— 一张表的乘法律

`torch.mm(a, b)` 这一个调用，背后可能走：

| 条件 | 走哪份实现 |
|---|---|
| a, b 在 **CPU** float32 | `mm_cpu` |
| 在 **CUDA** float32 | cuBLAS |
| 在 **CUDA** bf16 | 另一套 kernel |
| **稀疏**张量 | sparse mm |
| **`requires_grad=True`** | 先走 Autograd 版（记录计算图），再转到 CUDA 版 |
| 在 **`torch.compile` / FakeTensor** 里 | Meta 版：**只推 shape，不算数** |
| 在 **vmap** 里 | Batched 版 |

**7 种情况 × N 种后端 × N 种 dtype = 几十份实现**，而你在 Python 层只写了 `torch.mm` 一次。**dispatch 就是那个「看着输入、决定调哪一份」的路由器。**

### 2.3 机制：DispatchKey + DispatchKeySet

每个 tensor 身上携带一组 **DispatchKey**（本质是一个位集）。dispatcher 算出「最高优先级」的那个 key，查表拿到 kernel。

本机 `torch/include/ATen/core/dispatch/Dispatcher.h:785`，**一次算子调用的完整开销就这几行**：

```cpp
C10_ALWAYS_INLINE_UNLESS_MOBILE Return Dispatcher::call(
    const TypedOperatorHandle<Return(Args...)>& op,
    Args... args) const {
  auto dispatchKeySet =                                     // ← ① 从参数算出 dispatch key
      op.operatorDef_->op.dispatchKeyExtractor()
          .template getDispatchKeySetUnboxed<Args...>(args...);
  ...
  const KernelFunction& kernel =
      op.operatorDef_->op.lookup(dispatchKeySet);            // ← ② 查表拿到实现
#ifndef PYTORCH_DISABLE_PER_OP_PROFILING
  auto step_callbacks =
      at::getStepCallbacksUnlessEmpty(at::RecordScope::FUNCTION);
  if (C10_UNLIKELY(
          step_callbacks.has_value() && op.operatorDef_->op.isObserved())) {  // ← ③ profiler 开关
    return callWithDispatchKeySlowPath<Return, Args...>(...);
  }
#endif
  return kernel.template call<Return, Args...>(              // ← ④ 快路径：直接跳进去
      op, dispatchKeySet, std::forward<Args>(args)...);
}
```

本机 `c10/core/DispatchKey.h` 里**真实存在**的 key（节选）：

```text
CPU, CUDA, MkldnnCPU, Sparse, SparseCsr, NestedTensor,     // 后端
BackendSelect, Fake, Python,                               // 模式
Functionalize, Conjugate, Negative, ZeroTensor,            // 功能
ADInplaceOrView, Autograd,                                 // 自动微分
CompositeImplicitAutograd, CompositeExplicitAutograd,      // 组合
FuncTorchBatched                                           // vmap
```

⭐ 注意 `Python` 这个 key（`DispatchKey.h:230`）——**`TorchDispatchMode`、`FakeTensorMode` 就是挂在这里的**。这也是为什么 `torch.compile` 和各种「不改代码也能看见每个算子」的工具能插进来。

### 2.4 redispatch：一次算子调用为何走了两趟

§1.2 的 ①–④ 就是 `Dispatcher::call` 的四步：从参数算 key → 查表 → profiler 开关 → 跳进 kernel。这里只强调一个容易误解的点：

```text
Autograd 版（记录计算图）
   ↓  redispatch（★ 不重复记 RecordFunction，Dispatcher.h:848）
CUDA 版 → cuBLAS → 具体的 kernel 实现
```

一次 `aten::mm` 在 dispatcher 里其实**走了两趟**（Autograd 一趟、CUDA 一趟），但 profiler 里**只有一条事件**——因为 `Dispatcher.h:848` 明确写着 `// do not use RecordFunction on redispatch`。这是「事件数不会失控」的一个具体保护机制。

### 2.5 dispatch 层 vs kernel launch 层

| | **dispatch**（L0 记这里） | **kernel launch**（L1 的 uprobe 记这里） |
| --- | --- | --- |
| 记录单位 | **ATen 算子**（`aten::conv2d`） | **CUDA API 调用**（`cuLaunchKernelEx`） |
| 一次记录对应几个 launch | **0 个**（`view`/`size`/`empty`）、**1 个**（`add`）、**多个**（`conv2d` 经 cudnn 拆成多 kernel） | 恰好一个 kernel |
| 语义价值 | ⭐ **高**：能对应到你 Python 里写的那一行 | 低：你看到的是 `cutlass_80_tensorop_...` 这种名字 |
| 硬件对应 | 间接 | ⭐ **直接**：这就是 GPU 上实际跑的东西 |

- **L1（launch 层）** 能告诉你：「GPU 上一共跑了 N 个 kernel，第 37 个慢了 3 倍」；
- 但它**答不出**：「这 5 个 kernel 合起来，是你代码里的哪一次 `conv2d`？」
- **只有 dispatch 层（L0）有这个语义**——因为它是**按你代码的粒度**记的。

> 在 §1.2 / §4 的走位图里，L0 在 `Python` 与 `libtorch` 之间，L1 在 `libcudart` / `libcuda` 的宿主库边界；两层记录的是同一次执行的不同切面。

---

## 3. 运行时与驱动层：两半 + 一块固件

驱动是 OS 的一部分，OS 靠它把设备纳入自己的资源管理体系。**GPU 驱动几乎全在宿主侧**，由四块组成（前三块是「驱动」，第四块是驱动的消费者）：

### 3.1 内核态驱动 `nvidia.ko`（+ `nvidia-uvm.ko`、`nvidia-modeset.ko`）

真正碰硬件的部分：MMIO 寄存器、命令缓冲区组装与提交、runlist、中断、DMA、页表、UVM 页错误/预取/迁移。

**暴露给用户态的方式**是 `open/read/write/ioctl` + `mmap` on `/dev/nvidia*`、`/dev/nvidia-uvm`、`/dev/nvidiactl`。

### 3.2 用户态驱动 `libcuda.so`（+ `libnvidia-ptxjitcompiler.so`、`libnvidia-ml.so`）

CUDA **Driver API** 的实现，干大量不需要特权的事：**PTX→SASS 的 JIT 编译**、上下文/流管理、命令缓冲区组装、错误码映射。

Neutrino 论文就是从这个区分切入的：

> "Though drivers in OS are mostly referred to **kernel extensions exposed via read/write/ioctl syscall, such as `nvidia.ko`**, most vendors also maintain **higher-level user-space drivers as shared libraries such as `libcuda.so` or `libamdhip.so`** in Linux. These driver shared libraries are often highly complex and **closed-source**."

厂商的「驱动」在 Linux 上是**两层**——内核态扩展 + 用户态共享库。所以「驱动」不是一个层，是**用户态那一半 + 内核态那一半**。

### 3.3 GSP 固件（GPU System Processor）

跑在 GPU 板上一颗 **RISC-V 核**上，承担相当一部分传统上属于驱动的职责（初始化、部分调度与电源管理）。**这是唯一一块「驱动逻辑跑到设备侧」的东西**——但它不归 OS 管、不开源、不可插桩。

GSP 固件异常常被列为需求之一（「驱动 hang、CUDA context 卡死、MPS 多租户错误传播、固件（VBIOS/GSP）异常」）。

### 3.4 运行时/框架层 `libcudart.so` → `libtorch` → Python

**严格说这些不是驱动，是驱动的消费者。** `libtorch` 内部就是 §2 的 ATen dispatcher。

> ⚠️ 一个容易混的点：**CUDA runtime API（`cudaLaunchKernel`）不是 driver API（`cuLaunchKernel`）。** `libcudart` 是 CUDA Toolkit 里的便利层，内部才会调 `libcuda` 的 driver 入口。

---

## 4. 一次 kernel launch 的完整走位

把 §3 的四块串起来（上半段 Python → ATen dispatch → `cudaLaunchKernel` 见 §1.2 与 §2.4）：

```text
libcudart.so   cudaLaunchKernel()          ← 承接 §2.4 的 CUDA 版
  │
libcuda.so     cuLaunchKernel() / PTX JIT  ← L1 宿主库边界（uprobe 热点）
  │                                            ★ 驱动在这里有「用户态那一半」
═══ ioctl / mmap  on /dev/nvidia* ═══        ← ★ 用户/内核的缝
  │                                            （L1 的「OS 边界」就指这条缝）
nvidia.ko   分配显存 / 建页表 / 组装命令缓冲 / 写 GPU doorbell(MMIO)
nvidia-uvm  页错误 / 预取 / 驱逐 / 迁移     ← L2 驱动内 hook（struct_ops 挂这里）
  │                                            ★ 驱动在这里有「内核态那一半」
═══ PCIe / NVLink / 命令队列 / runlist ═══    ← ★ 宿主/设备的物理缝
  │
GPU：GigaThread 引擎取 runlist → 分 block 到 SM → 硬件 warp 调度器
  │
执行 SASS 指令流                             ← L3 kernel 内插桩
  │
L1/L2 cache、HBM、ECC 逻辑、PMU 计数器       ← L4 硬件计数器（只读）
```

**结论**：从 `libcudart` 往下，一次 launch 就进入「用户态 `libcuda.so` → `ioctl`/`mmap` → 内核态 `nvidia.ko` / `nvidia-uvm.ko` → 命令队列/runlist → GPU」这条链路；`libcuda.so` 之上是 §2 的算子层，之下才是真正碰硬件的部分。

---

## 5. FatBinary：编译产物容器

### 5.1 它装了什么

同一份 `kernel<<<>>>`，nvcc 会**同时对多个目标架构**编译，产物全部塞进一个文件里：

```text
宿主 ELF（可执行文件 / .so）
  └─ .nv_fatbin 段                        ← fatbin 就嵌在这里
       └─ FatBinary 容器
            ├─ cubin (sm_80)   ← SASS 机器码，特定架构
            ├─ cubin (sm_89)   ← SASS
            ├─ cubin (sm_90)   ← SASS
            ├─ PTX (compute_80)  ← 文本 IR，向前兼容用
            ├─ PTX (compute_90)
            └─ NVVM IR / LTO IR  ← 只在 -rdc=true（分离编译）时出现
```

**为什么要「胖」**：因为你编译时不知道最终会跑在哪块卡上。

- **有匹配架构的 cubin** → 直接用（最快，零编译）；
- **没有匹配的 cubin 但有 PTX** → 驱动**当场 JIT** 把 PTX 编成目标架构的 SASS（这就是 §3.2 提到的 `libnvidia-ptxjitcompiler.so`）。

于是形成一条**代码形态链**：

```text
CUDA C++  →  PTX（文本 IR）  →  SASS（机器码，绑架构）
   编译期        架构中立          架构相关
```

### 5.2 别混淆的三个同名物体

| 名字 | 是什么 |
|---|---|
| **FatBinary** | 容器格式（本节的回答对象） |
| `__cudaRegisterFatBinary(void *fatCubin)` | Runtime 的**登记函数**，参数是 fatbin 映像指针 |
| `cuModuleLoadFatBinary` / `cuModuleLoadData` | Driver 的**加载函数**，喂给它的也是 fatbin/cubin/PTX 字节 |

---

## 6. 登记 vs 加载：`__cudaRegisterFatBinary` 与 `cuModuleLoad`

**核心关系：`__cudaRegisterFatBinary` 是「登记」，`cuModuleLoad` 是「加载」，中间隔着一个 lazy loading 阶段。** 而 **fatbin 就是在这两者之间流通的「货币」**。

### 6.1 三个阶段

| 阶段 | 谁在做 | 发生了什么 | 碰 GPU 了吗 |
|---|---|---|---|
| **① 编译期打包** | `nvcc` | 生成 SASS + PTX → 打包成 fatbin → 嵌进宿主 ELF 的 `.nv_fatbin` 段 | ❌ |
| **② 登记期** | **CUDA Runtime**（`libcudart`，通常静态链进你的库） | 库被加载时（程序启动或 `dlopen`）**静态初始化器**逐个调用 `__cudaRegisterFatBinary()` → runtime **记录**这个映像；每个 `__global__` 再调 `__cudaRegisterFunction()` 建立 host stub ↔ device function 映射 | ❌ **只是记账** |
| **③ 加载期** | **CUDA Driver**（`libcuda.so`） | 首次使用该 kernel 时，驱动从 fatbin 里**挑最合适的一份**（匹配的 cubin，或 JIT PTX）→ 创建 **module 对象**（占显存、建页表）→ 得到 function handle | ✅ **真正上 GPU** |

> 「登记 / 首次 launch / 稳态 launch」是**三条不同时间线**，下面分开画：§6.2 只登记（库加载时），§6.3 是首次 launch（含真正的 lazy 加载），§6.4 是之后每次 launch。

### 6.2 登记期走位（库加载时，只记账、不碰 GPU）

```text
my_kernel<<<...>>>()                       ← 你写的东西（host stub 已在编译期生成）
  │
libcudart.so（CUDA Runtime）· 静态初始化器（程序启动 / dlopen 时跑）
  │   ├─ __cudaRegisterFatBinary(&fatbin_wrapper)
  │   │      └─ runtime 登记映像（不加载！只记账）
  │   └─ __cudaRegisterFunction(handle, "my_kernel", ...)
  │          └─ 建立 host stub ↔ device function 映射
  │
  ✗ 到此为止：只填表，连 libcuda 都没进
```

### 6.3 首次 launch 走位（含 lazy 加载，Runtime/Driver 交界）

```text
my_kernel<<<...>>>()                       ← host stub 被调用
  │
libcudart.so（CUDA Runtime）
  │   └─ cudaLaunchKernel(...)             ← 拿 host stub 地址查 §6.2 登记表
  │          └─ [首次·LAZY] 内部请求驱动创建 module   ★ 真正的加载点
  │
═══ Runtime / Driver 的交界 ═══
  │
libcuda.so（CUDA Driver）
  │   ├─ cuModuleLoad*        ← 用 fatbin/cubin/PTX 字节创建 module（占显存 / 建页表）
  │   ├─ cuModuleGetFunction  ← 取 CUfunction（注意：它本身也会触发加载，见 §7）
  │   └─ cuLaunchKernel       ← launch
  │
═══ ioctl / mmap on /dev/nvidia* ═══
  │
nvidia.ko + nvidia-uvm.ko → 命令队列/工作描述符 → GPU
```

### 6.4 稳态 launch 走位（第 N 次，N ≥ 2）

```text
my_kernel<<<...>>>()
  │
libcudart.so  cudaLaunchKernel()   ← 查表：module handle / CUfunction 已缓存，不再 load
  │
libcuda.so    cuLaunchKernel()     ← 直接拿已有 CUfunction 下发
  │
═══ ioctl / mmap on /dev/nvidia* ═══
  │
nvidia.ko + nvidia-uvm.ko → 命令队列/工作描述符 → GPU
```

（相比 §6.3，稳态这条**没有 `__cudaRegister*`，没有 `cuModuleLoad`，也没有 `cuModuleGetFunction`**。）

---

## 7. lazy loading

在程序启动时不会立即加载所有 CUDA 模块，而是推迟到真正需要使用时才加载，从而缩短程序的初始化时间。两个要点：

1. **`cuModuleGetFunction()` 会导致模块被加载到 device memory。** → 所以 `cuModuleGetFunction` **不只是查询，它本身就是加载触发点**。
2. **"`cuModuleLoad()` does not guarantee that a module will be loaded immediately."** → 你 hook 了 `cuModuleLoad` 也可能没赶上真正的加载时刻。

> 这也是「想 hook 加载点却抓不到」的根源——**登记、`cuModuleLoad`、`cuModuleGetFunction` 不是同一个时刻**（与 [04](./04_性能分析方法论.md) 的 profiling 插桩选择相关）。

---

## 8. 用户态 Hook：LD_PRELOAD

`LD_PRELOAD` 是 Linux（以及类 Unix 系统）下的一个**环境变量**，用于在程序启动时**优先加载指定的共享库**，让这些库中的函数符号先于系统默认库被解析。它是用户态做「拦截/Hook」的经典机制。

### 8.1 工作原理

Linux 动态链接器（`ld.so`）在解析程序依赖的符号时，遵循这样的查找顺序：

1. `LD_PRELOAD` 指定的库（最高优先级）
2. 程序自身链接时声明的库（DT_NEEDED）
3. 系统默认路径（如 `/lib`、`/usr/lib`）中的库

也就是说，如果你在 `LD_PRELOAD` 中指定了一个自定义库，里面定义了与某个系统函数**同名同签名**的函数（比如 `malloc`、`cuInit`），那么程序实际调用的就是你的版本。

### 8.2 典型用法

```bash
# 运行 myapp 时优先加载 ./myhooks.so
LD_PRELOAD=./myhooks.so ./myapp
```

配合 `dlfcn` 中的 `dlopen` / `dlsym`（常用 `RTLD_NEXT` 找到「真正的下一个同名函数」），可以在自己实现的函数里先做点额外工作，再转调原始函数——这就是论文中 **NEUTRINO hook driver** 的做法。

### 8.3 为什么和驱动/插桩有关

§4 那张走位图里，`libcuda.so`（用户态驱动那一半）正是 `uprobe`/`LD_PRELOAD` 最容易命中的热点；而内核态那一半要靠 `struct_ops` 之类挂点。用 `LD_PRELOAD` 拦截 `libcuda`/`libcudart` 的入口，就能在不改框架、不碰 kernel 的情况下旁路记录调用——这与 PaperReadingNotes 里 Fail-slow 笔记用 `LD_PRELOAD` hook NCCL 顶层接口是同一套手法。

---

## 9. 分层可观测性：L0 与 L1 怎么配合

§1.1 给了 L0–L4 的地图，§2.5 给了 L0/L1 的静态分工；真正采集时，两层要一起用：

- **两层必须配合**：只有 L1 会看到一堆 `cutlass_*` kernel 名，无法回溯到代码行为；只有 L0 则看不到 kernel 级的实际耗时与硬件计数。
- **事件数不会爆炸**：`redispatch` 明确不重复记 `RecordFunction`，所以 Autograd + CUDA 两趟派发在 profiler 里仍只有一条事件。
- **语义映射的方向**：L0 回答「这段代码调了哪些算子、各几次」，L1 回答「GPU 上实际跑了哪些 kernel、各多久」；把两者按时间轴对齐，才能既定位到代码行、又定位到硬件瓶颈。

一句话：**L0 告诉你「代码在干什么」，L1 告诉你「GPU 在跑什么」**；把时间轴对齐，才能从 `cutlass_*` 回溯到 `aten::conv2d`，再从算子耗时下钻到具体 kernel。
