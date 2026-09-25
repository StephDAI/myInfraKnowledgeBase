## 一、问题场景
### 计算类
出现概率低、短命、影响温和
**两个案例**：
- **Case-1 CPU 争抢**：吞吐骤降 21.6%，4 张卡 SM 利用率同步下滑，看似 GPU 变慢；暂停做矩阵测试却发现 GPU 性能正常。真正的根因是同节点高 CPU 作业数激增，导致训练作业 CPU 满足率下降——CPU 喂不饱数据/GPU，吞吐掉下来。
- **Case-2 GPU 性能退化**：前 10 分钟变慢，进一步定位发现只有 GPU0 比其它卡慢 20%，原因是**温度节流（thermal throttling）**。但升温不必然导致降速，更可能是硬件问题，发生率约 0.5%
### 通信类
出现频率高、持续久、影响大，主要为网络拥塞，**没有发现硬件（如 RNIC）导致的变慢**。多项作业共享同一链路，导致触发RoCE拥塞控制压制带宽。
### 大规模场景
作者人工检查了一个月内 27 个 512–1024 GPU 的大作业：
- **16/27 出现 fail-slow，平均拖慢 JCT 34.59%**，20% 的作业被拖慢超过 50%——比小规模严重得多；
- 平均持续时间 72 分钟，远长于探针测得的小规模时长；
- 原因仍以网络拥塞为主（13 个），其余是"网络 + GPU 退化"复合问题；大作业独占整机 8 卡、不与别人共置，所以**没有 CPU 争抢**
**启示**：大规模下计算与通信 fail-slow 会**复合出现**，缓解策略必须自适应、灵活，能动态处理复杂组合问题。
## 二、Greyhound-Detect
设计目标四条：非侵入且框架无关（R1）、快速准确（R2）、全自动（R3）、轻量（R4）
采用master-worker架构
![[Pasted image 20260925160518.png]]
#### 阶段 1：Tracking（零打扰，常开）
Monitor 用 `LD_PRELOAD` hook NCCL 顶层集合通信接口（AllReduce/AllGather/ReduceScatter 等），只记录调用类型和时间戳，不改框架、不碰 CUDA kernel。由于只拦截顶层接口，对 ACCL、MSCCL 或定制 NCCL 同样适用。这里有两个巧妙的时间序列技巧：

- **ACF 自动发现 iteration 周期**：不同框架/并行策略下每个 iteration 的通信调用序列不同且事先未知，作者对调用序列做自相关分析，取第一个 ACF ≥ 0.95 的 lag 作为周期，从而反推出 iteration 时间。
- **BOCD + 验证**：用贝叶斯在线变点检测在线性时间内找 iteration 时长的突变点；但裸 BOCD 误报率高达 18–34%（Python GC、CUDA allocator cache miss、变长序列都会造成抖动），所以加了一步验证——变点前后平均 iteration 时间差 <10% 就判为抖动，防止突发抖动误报，只有持续退化才报Fail-slow。这个"BOCD+V"组合在计算类 fail-slow 上做到 100% 准确、0% FPR/FNR，通信类 99.1% 准确。
worker找出问题，由LocalAnalyzer上报给GlobalController
### 阶段 2：Profiling（轻量缩小嫌疑范围）
检测到慢 iteration 后，给 NCCL 调用注入 CUDA event 测各通信组的执行时间，然后 GlobalAnalyzer 做**跨组比较**：把传输量相同的通信组（如同样处理一批 DP 的 all-reduce 组）聚成 comparable cluster，执行时间超过中位数 10% 的组即为嫌疑组。这一步把"全集群 benchmark"缩成"几个嫌疑组"。
### 阶段 3：Validation（短暂挂起，精确定位）
通过 hook 住的 NCCL 调用把训练"陷"进等待循环实现**免 checkpoint 的挂起**，然后只在嫌疑组内跑 benchmark：计算侧跑 FP8/FP16/FP32 GEMM；通信侧不枚举全部链路（O(N²)），而是把 ring/tree 集合通信拓扑拆成不重叠的 P2P 对，ring 2–3 趟、tree 4 趟并行打完，**O(1) 与组大小无关**。

当前的设计无法检测计算和通信内核以特定模式共同执行时才会出现的fail-slow。但该种情况出现频率极低。

## 三、Greyhound-Mitigate 故障缓解
随解决方案效果增强，成本也增加。需设计选择策略。当累积减速等于该策略的动作开销时，它切换到下一个策略。
### 方法一：不作为等自愈
### 方法二：调 micro-batch 分配
按各 DP 组实测速度重新分配 micro-batch 数，可有效应对计算fail-slow问题。并使用加权梯度聚合方法保持训练损失的一致性。
### 方法三： 调整并行拓扑
计算 & 通信 fail-slow 均能缓解。两个抓手——(a) DP 流量（每 iteration 数十 GB 梯度）远大于 PP 流量（几十到几百 MB 激活值），把拥塞链路从 DP 组换到 PP 组。
(b) 一个PP stage中有一组GPU，一个stage处理一个micro-batch的速度取决于最慢的GPU。若分散在两个 stage 则延迟叠加（论文例子：8.5s → 8s），所以把 straggler 尽量合并进最少 PP stage、且避开首尾 stage。实现上是暂停训练 → 参数拷贝到 host DRAM 主存 → 参数通过 RDMA P2P 互换 → 恢复，消耗中等时间。
### 方法四：检查点 & 重启
在健康节点上重启训练