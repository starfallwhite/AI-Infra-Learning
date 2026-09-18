# AI Infra 训推加速学习计划（C++ 校招生版）

> 目标：用 **24 周、每周 15～20 小时**，形成可投递训练/推理引擎、算子优化、异构计算岗位的项目与知识体系。

## 1. 岗位要求归纳

检索关键词：`AI Infra`、`训练系统`、`推理引擎`、`高性能计算`、`CUDA/算子优化`、`分布式训练`、`编译器`。

| 能力层 | 常见要求 | 我的优先级 |
|---|---|---|
| 编程基础 | 熟练 C/C++、数据结构与算法、Linux；能定位并发/内存/性能问题 | P0 |
| GPU 与算子 | CUDA、GPU 存储层次、线程模型、矩阵乘；会用 Nsight 做 profiling | P0 |
| 推理加速 | Transformer/LLM 基础，KV Cache、量化、连续批处理；了解 TensorRT-LLM、vLLM | P0 |
| 训练加速 | PyTorch，数据并行/张量并行/流水线并行，混合精度、通信与显存优化 | P1 |
| 分布式系统 | NCCL、集合通信、网络基础；理解吞吐、延迟、扩展效率和容错 | P1 |
| 工程素质 | Python 能读写实验脚本，CMake/Git/测试/Benchmark，阅读英文文档与论文 | P0 |
| 加分项 | Triton、CUTLASS、MLIR/LLVM、RDMA、开源贡献 | P2（选一个深入） |

**投递策略：** C++ 是优势，不必先转“算法岗”；主攻“推理引擎/算子优化”，补齐 Python + PyTorch，再扩展分布式训练。每周收集 5 个 JD，将关键词、要求频次和自身证据记录下来，每月调整一次优先级。

## 2. 24 周路线

| 阶段 | 周数 | 学什么 | 可验收产出 |
|---|---:|---|---|
| 系统基本功 | 1～4 | C++17/20、RAII/并发/SIMD；Linux、CMake、GDB、perf；Python/PyTorch 基础 | C++ 线程池 + 单测 + benchmark；完成一个 PyTorch MLP 训练脚本 |
| CUDA 算子 | 5～9 | 执行/内存模型、访存合并、shared memory、reduction、GEMM；Nsight | naive→tiled GEMM、softmax、layernorm；给出正确性测试及优化前后数据 |
| 模型与推理 | 10～14 | Transformer、attention、KV Cache；batching、量化、算子融合 | 用 C++/CUDA 做 mini Transformer 推理；记录首 Token 延迟、Token 吞吐和显存 |
| 分布式训练 | 15～18 | DDP、NCCL collective、ZeRO；混合精度、梯度累积/checkpoint | 2+ GPU（无卡则用云短租）训练实验；比较单卡/多卡吞吐与扩展效率 |
| 框架源码 | 19～21 | 精读 PyTorch dispatcher/autograd 或 vLLM 调度/KV Cache，二选一 | 一篇源码导读；修复 issue、补测试或提交小 PR |
| 求职冲刺 | 22～24 | 整理项目、系统设计、八股与算法题；针对 JD 模拟面试 | 2 个主项目、技术简历、项目讲解稿、至少 4 次模拟面试 |

## 3. 每周执行模板

- **工作日（每天 2 小时）：** 45 分钟理论 + 60 分钟编码 + 15 分钟笔记。
- **周末（5～8 小时）：** 完成一个可运行里程碑，profiling，并更新 README/图表。
- **周复盘：** 我优化了什么？瓶颈证据是什么？指标提升多少？下周只选一个主目标。
- 时间不足时依次保留：**项目 > C++/CUDA > 模型原理 > 算法题 > 泛读论文**。

## 4. 作品集标准

### 项目 A：CUDA 算子库（必做）

- GEMM、softmax、layernorm/attention 至少 3 个算子；CPU/PyTorch 结果作基准。
- 覆盖不同 shape、dtype 和边界输入；提供单测、benchmark、Nsight 截图。
- README 解释瓶颈、优化过程和硬件环境，不只写“提升了 N 倍”。

### 项目 B：迷你 LLM 推理引擎（必做）

- C++ 主体，支持权重加载、token-by-token decode、KV Cache 和 batch 推理。
- 至少实现一种优化：FP16/INT8、算子融合或连续批处理。
- 指标包含 TTFT、TPOT、tokens/s、峰值显存，并与一个成熟框架同环境对比。

### 项目 C：分布式训练实验（选做）

- 比较 DDP 与 ZeRO/不同并行策略，记录通信占比、吞吐和扩展效率。
- 能解释为何没有线性加速，以及 compute/communication overlap 的改进方向。

## 5. 面试检查清单

- [ ] 能手写常见数据结构，并解释 C++ 对象生命周期、智能指针、移动语义和并发安全。
- [ ] 能从 occupancy、访存、分支和算术强度分析 CUDA kernel，而非只会调用 API。
- [ ] 能讲清 prefill/decode、KV Cache、量化、batching 对延迟/吞吐/显存的影响。
- [ ] 能讲清 all-reduce、DP/TP/PP、ZeRO 以及训练显存的组成。
- [ ] 每个项目能按“问题—基线—profiling—优化—指标—局限”在 5 分钟内讲完。
- [ ] 保持每周 3～5 道算法题，但不挤占项目时间。

## 6. 一手学习资料

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- [PyTorch Distributed Overview](https://docs.pytorch.org/tutorials/beginner/dist_overview.html)
- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)
- [vLLM Documentation](https://docs.vllm.ai/)
- [TensorRT-LLM Documentation](https://nvidia.github.io/TensorRT-LLM/)
- [CUTLASS Documentation](https://docs.nvidia.com/cutlass/)

## 7. 今天就开始

1. 建立 `cpp-basics` 与 `cuda-kernels` 两个目录，配好 CMake、GoogleTest 和 benchmark。
2. 选 20 个目标岗位建表统计关键词，确定主投“推理引擎”或“算子优化”。
3. 本周完成 CPU 向量加法与 CUDA 向量加法，验证正确性并记录带宽和耗时。






C++ / CUDA 大模型推理加速工程师 → 再扩展到训练加速与分布式系统。

目前 NVIDIA、百度等相关岗位的共同要求非常集中：C++/Python、CUDA、GPU 架构、PyTorch、LLM 推理机制、性能分析，以及 vLLM/SGLang/TensorRT-LLM；更深一层是 CUTLASS/Triton、NCCL 和分布式训练/推理。

你需要的能力树
优先级	学什么	学到什么程度
★★★★★	C++	现代 C++、内存、并发、性能优化
★★★★★	CUDA	能自己写 Kernel + 优化
★★★★★	GPU 架构	SM / Warp / Shared Memory / Tensor Core
★★★★☆	PyTorch	会用只是基础，要理解 Tensor / Operator / CUDA 调用链
★★★★★	Transformer 推理	Attention、KV Cache、Prefill/Decode、Batching
★★★★☆	推理框架	vLLM / SGLang / TensorRT-LLM 至少深入一个
★★★★☆	Profiling	Nsight Systems / Nsight Compute
★★★☆☆	分布式	NCCL、TP/DP/PP/EP、通信计算 Overlap
★★★☆☆	训练加速	DDP/FSDP/ZeRO/Megatron
★★☆☆☆	编译优化	Triton / CUTLASS / CuTe / torch.compile

百度当前校招 AI Infra 就直接要求 C++/Python、CUDA、分布式、混合精度、量化、KV Cache、算子融合；更偏底层的岗位进一步要求 CUDA/CUTLASS/CuTe/Triton/TileLang、vLLM/SGLang/TensorRT-LLM。

NVIDIA 的 New Grad 推理岗要求也几乎一致，而且特别强调 性能 profiling、GPU 架构、CUDA kernel、vLLM/SGLang/TensorRT-LLM。

我建议你的学习路线

不要按照“深度学习 → Transformer → CUDA → 分布式 → vLLM”平均用力。你是 C++ 工程背景，应该走 系统性能路线：

第一阶段：2～3 周

C++ 性能基础
↓
Linux
↓
CPU Cache / SIMD / 多线程
↓
计算机体系结构

重点补：Cache、虚拟内存、NUMA、SIMD、线程同步、memory bandwidth、compute bound / memory bound。

你现在已有 C++ 基础，这部分不用重新学语言。

第二阶段：4～6 周，最重要

把主要时间投入 CUDA：

Thread / Block / Grid
        ↓
Warp / SM
        ↓
Global / Shared / Register
        ↓
Coalesced Memory Access
        ↓
Reduction
        ↓
GEMM
        ↓
Softmax / LayerNorm
        ↓
Kernel Fusion
        ↓
Tensor Core

自己实现：

vector_add
reduction
softmax
layernorm
matmul
fused kernel

然后用：

Nsight Systems
Nsight Compute

回答一个核心问题：

我的 Kernel 为什么慢？

而不是只做到“CUDA 程序能运行”。

第三阶段：3～4 周

补 LLM 推理原理：

Transformer
   ↓
Attention
   ↓
MHA / MQA / GQA
   ↓
KV Cache
   ↓
Prefill / Decode
   ↓
Continuous Batching
   ↓
PagedAttention
   ↓
Quantization FP16/FP8/INT8/INT4
   ↓
Speculative Decoding

尤其要理解：

Prefill → Compute Bound
Decode  → Memory Bound

以及：

吞吐量
Latency
TTFT
TPOT
KV Cache 显存占用

这些现在就是推理 Infra 面试高频核心。

第四阶段：4 周

直接读 vLLM，不要同时学五个框架。

路线：

跑通 vLLM
    ↓
理解 Scheduler
    ↓
理解 KV Cache
    ↓
理解 PagedAttention
    ↓
理解 Continuous Batching
    ↓
理解 Worker / ModelRunner
    ↓
看 CUDA/Triton Attention Kernel

然后再横向了解：

vLLM
SGLang
TensorRT-LLM

目前招聘也大量直接点名这些框架。

第五阶段：3～4 周

开始进入真正的“训推”：

单 GPU
  ↓
多 GPU
  ↓
NCCL
  ↓
TP
DP
PP
EP
  ↓
通信 / 计算 overlap

训练侧学习：

PyTorch DDP
↓
FSDP / ZeRO
↓
Megatron-LM
↓
Transformer 并行训练

这也是当前训练 Infra 岗位重点，例如千卡训练、NCCL、TP/DP/PP/EP 和通信优化。

最关键：项目怎么做

校招生不要做：

“基于 vLLM 部署 Qwen。”

这个项目区分度非常低。

建议你的简历最后形成 两个项目。

项目一：CUDA 算子优化

例如：

Transformer 高性能 CUDA 算子库

实现：

Softmax
LayerNorm / RMSNorm
GEMM
Attention
RoPE
Fused Kernel

优化：

Shared Memory
Vectorized Load
Warp Primitive
Tensor Core
Kernel Fusion

最后一定有：

Baseline: PyTorch
自己的 CUDA Kernel
Latency
Bandwidth
TFLOPS
Speedup
Nsight 分析

这非常符合 C++ 校招生的定位。

项目二：Mini LLM Inference Engine

建议：

C++ + CUDA
        ↓
加载 Qwen/Llama
        ↓
Tokenizer
        ↓
Transformer Forward
        ↓
KV Cache
        ↓
Continuous Batching
        ↓
CUDA Kernel
        ↓
NCCL Tensor Parallel

做到：

自己实现一个“小型 vLLM”。

这个项目的价值会远大于几个普通深度学习项目。

你的学习时间分配

如果你现在开始准备，我建议：

C++ / 系统基础        10%
CUDA / GPU            30%
LLM 推理原理          15%
vLLM / SGLang         20%
分布式 / NCCL         15%
训练框架               10%

也就是说：

不要把大量时间花在模型训练、调参、Prompt、LangChain 上。

你的竞争力应该是：

别人：
Python + PyTorch + LLM

你：
C++
+ CUDA
+ GPU Architecture
+ PyTorch Runtime
+ LLM Inference
+ vLLM
+ NCCL
+ Performance Optimization

这是目前 AI Infra 岗位非常典型的一条技术栈。NVIDIA 甚至明确把 CUDA kernel、CUTLASS/Triton、PTX/SASS、Nsight、vLLM/SGLang/TensorRT-LLM 作为核心或加分能力。

我给你的最终路线
C++基础
  ↓
计算机体系结构
  ↓
CUDA ★★★★★
  ↓
GPU性能分析
  ↓
Transformer推理原理
  ↓
手写Transformer CUDA算子
  ↓
vLLM源码
  ↓
Mini LLM Inference Engine
  ↓
NCCL + TP/DP/PP
  ↓
Megatron / 分布式训练
  ↓
Triton / CUTLASS / 编译优化

对于你这种 C++ 主栈的应届生，我认为最值得先冲的是 “LLM 推理引擎 / CUDA 性能优化 / AI Infra 系统研发”，而不是先从大规模训练平台入手。前者更容易利用你的已有优势，也更容易通过个人 GPU 项目做出可量化的成果。