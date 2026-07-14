# MagiAttention × Megatron Hybrid MLA/DSA 研发与黄金 Benchmark 方案

状态：架构方向与价值假设正在冻结（2026-07-14）。T00–T02 已完成；当前暂停在 T03 开始前，只完善方案，不继续源码集成和 benchmark 实现。

## 太长不看

项目不再创建独立的 LongAttention runtime，而是在 MagiAttention 上迭代。MagiAttention 的稳定边界是 **CP layout + distributed attention execution**；Megatron 继续拥有完整 MLA/DSA 模型语义、所有参数、RoPE、indexer loss、checkpoint 和层生命周期。完整 MLA E2E 由 `Megatron semantic preparation -> typed execution request -> Magi distributed execution -> Megatron finalization` 组合完成，不要求 Magi 重新实现一套 MLA layer。

第一核心要素不是立即优化 kernel，而是先搭出稳定的整体骨架：统一 case/schema、global batch、Megatron native/Magi adapters、两条端到端 harness、MLA+DSA extension seam、Hybrid recipe、provenance 和测试矩阵。研发优先级冻结为：**整体骨架 > expanded dense MLA 必要基线 > absorbed MLA+DSA 核心能力 > 3:1 Hybrid 组合验证 > 内部 KDA/Linear 优化**。第一版 Hybrid recipe 固定为每 4 层一个单元：`3×Linear/KDA proxy + 1×MLA`；后训练时该 MLA 层升级为 `MLA+DSA`。Linear 层先使用 Megatron GatedDeltaNet 作为占位，只证明生命周期和组合正确，不投入性能优化。所有性能结论都必须同时通过正确性、端到端延迟、显存和通信量四个门。

MLA 开发基线采用 MagiAttention PR #331；PR #326 仅作为历史实现和失败案例参考。#331 明确 supersede #326，不能把两套补丁叠加。两者提供的是 expanded Q/K/V 的非对称 head-dim distributed-attention 能力，不是 Megatron 模型级 MLA projection、absorbed MLA 或 DSA 的完整实现。

NVIDIA NeMo Automodel PR #2384 作为第三份训练集成参考。它验证了 `setup -> packed batch -> Magi key/layout -> dispatch inputs/labels/positions -> registered attention -> local loss -> CP token/loss reduction` 这条端到端生命周期，支持我们把 benchmark 起点放在 batch 生成/packing 之前，而不是从预生成 Q/K/V 开始。但它当前显式拒绝 `QK dim != V dim`，不支持 MLA，更不包含 absorbed MLA、DSA indexer 或 exact distributed top-k；因此只借鉴集成结构、packed metadata 与测试门禁，不能作为本项目 MLA/DSA 能力或性能 oracle。

Megatron dynamic/hybrid CP 与 Magi 不解决同一个层次的问题。dynamic CP 是**外层资源调度**：根据样本长度选择本次使用的 CP degree/group，减少短序列被过度切分；Magi 是**选定 CP group 后的组内 attention execution**：决定 token/chunk layout、KV/O 通信、kernel 和 overlap。推荐组合是 `dynamic CP scheduler -> Magi layout/runtime -> MLA/DSA execution`，而不是二选一。首轮价值实验仍先固定 CP，隔离并证明 Magi 的组内净收益；之后才加入 dynamic CP 做长度异构 workload 的组合实验。

## 0. 执行边界与当前状态

- 当前先完善和评审本方案；在本节新增的 dynamic CP 价值判断冻结前，不继续修改源码或实现 benchmark。
- 本地与远端 MagiAttention 已经对齐到 `magic@529fb0a4e273b3557a56d8afd60b74da46688095`；这是此前已经发生的状态，不代表后续执行已获授权。
- 当前工作树中的 4 个已修改测试/kernel 文件、`.unisonignore` 和其他用户改动都视为用户资产；后续执行前先保存路径、状态和哈希，任何补丁不得覆盖它们。
- PR #331 已在隔离远端 worktree 做 preflight，未合入活动 `magic`；项目内唯一环境 `/root/work/MagiAttention/.venv` 已建立。PR #331 exact head 的 FA4 `Q/K=192,V=128`、BF16、GQA 8:2、12K varlen forward/backward parity 已通过 CP1 与 CP4；这些是 correctness preflight，不是性能数字。benchmark 代码和正式测试矩阵尚未实现。

## 1. 已冻结的决策

- 产品与代码主仓库：`MagiAttention`。
- 本地/远端开发分支：均为 `magic`，跟踪 `origin/magic`。
- MagiAttention 基础提交：`529fb0a4e273b3557a56d8afd60b74da46688095`。
- MLA patch：PR #331，提交 `99f836e852990579363a366bf1276d828c3e53d4`。
- Megatron baseline：`779c5b748dbcf00ad9e36d539c576b404ab4abe9`。
- 远端：`aistudio-61480188-ssctl`，用户指定为 4×B200；精确 GPU 名称、UUID、SM、显存和 NVLink 拓扑在 Gate 0 重新取证后写入 provenance。
- 数据类型：第一版只使用 BF16；FP8、FP4 和 CUDA Graph 不进入首轮价值判断。
- Mask：第一版只支持 full causal 和 packed-varlen causal，不追求 Magi 的任意 mask 全覆盖。
- KDA：第一阶段使用 Megatron `GatedDeltaNet` 作为结构代理，不将其结果表述为真实 KDA 性能。
- Hybrid 层配比：每个 4-layer unit 固定为 `LinearProxy, LinearProxy, LinearProxy, MLA`，即 3:1；多个 unit 按同一 pattern 重复。后训练只把 unit 中的 MLA execution 升级为 MLA+DSA，Linear/KDA 层比例不变。
- 术语映射：本文将用户所说的 3 个 Hybrid/Linear slots 暂记为 `LinearProxy`，S0 使用 GDN 实例化；后续替换内部 KDA 时保持 slot 数量和顺序，不把 GDN 的具体数学结构写进通用 contract。
- 第一优先级：搭骨架而非优化 Linear。P0 只要求 LinearProxy 能参与 forward/backward、recompute、PP、checkpoint 和 layout round-trip；其 kernel 性能、state scan 和 persistent-layout 优化全部后置。
- 功能优先级：expanded dense MLA 只作为 PR #331 接通与 native golden 的必要前置；随后立即实现并验证 absorbed MLA+DSA。完整 3:1 Hybrid 训练生命周期排在 MLA+DSA 之后。S0 中的 Hybrid 只做 schema/build/CP1 smoke，不能据此提前启动 Hybrid 优化。
- DSA 语义：DSA 不替换 MLA 参数化。`MLA+DSA` 保留 MLA projection、latent KV、RoPE、checkpoint 和 output projection，在其上增加 indexer、exact top-k 与 sparse attention execution；不存在独立的“DSA attention 分支”。
- 框架边界：Megatron 是 model-semantic owner；MagiAttention 是 distributed-execution owner；`MegatronMagiAdapter` 只负责把前者产生的 tensor/metadata 变成类型化执行请求，并把结果交回前者完成 loss/state/output finalization。
- 表示与选择正交：`expanded_qkv | absorbed_latent` 描述 attention 执行看到的表示，`dense | exact_topk` 描述 token 选择方式。PR #331 首先覆盖 `expanded_qkv+dense`；最终 MLA+DSA 目标是 `absorbed_latent+exact_topk`，不能用 expanded sparse 路径冒充。
- 参数所有权：Q/KV LoRA、norm、RoPE、DSA indexer、`up_v_weight`、output projection 等参数始终由 Megatron ModuleSpec 注册和保存。Magi backend 只能借用带 autograd 的 tensor/weight view，不得重新注册参数或改变 state-dict keys。
- Layout 与执行分离：`CPLayoutProvider` 决定一个 microbatch 的 physical token layout；`AttentionExecutionBackend` 在该 layout 上执行 expanded/absorbed、dense/sparse attention。二者可以组合，但不能合并成一个依赖 Megatron 私有 forward 的大对象。
- Layout：分阶段推进。先实现 canonical storage layout + attention-local dispatch/combine，建立无争议 golden；只有在证明 KDA/GDN 的有序 recurrent state 能与目标 layout 正确协作后，才研究跨 GDN/MLA 层持久 layout。
- Benchmark 形状：先公开可复现的 DeepSeek-like 形状，最终价值门必须换成内部真实配置。
- Native baseline：dense MLA 对每个 case 枚举 Megatron/TE 实际支持的 `p2p`、`all_gather`、`a2a` 和可用的 hierarchical 方式，headline 使用其中最快且正确的结果；MLA+DSA 的当前 native CP oracle 使用其唯一支持的 `allgather`。
- 现有 `/Users/fankun/Documents/LongAttention` 不再作为产品仓库；其中 runner、typed contract 和 Gate 0 测试只作为迁移输入，不在两处继续演进。
- Automodel #2384 定位：训练集成与 capability-gate 参考，不是 Megatron-native 接口、MLA/DSA correctness oracle 或性能 baseline。可复用思想必须重新表达为本项目显式的 `MagiIntegration + CPRuntimeContext + ModuleSpec/ProcessGroupCollection`，不得复制其进程全局 recent-key 状态。

### 1.1 Dynamic CP 与 Magi 的职责边界

先固定两个定义，避免后续把“动态选择多少张卡”和“这些卡如何做 attention”混为一谈。

| 层次 | 当前 Megatron dynamic/hybrid CP | MagiAttention |
| --- | --- | --- |
| 主要输入 | 每个样本的 sequence length、每 rank 最大承载长度、DP×CP rank pool | 已选定的 CP group、全局 token/mask/packed metadata、Q/K/V 或 latent execution request |
| 主要决策 | 为样本选择 `local_cp_size`，把样本分配给预创建的 power-of-two DP×CP subgroup | 在 group 内规划 token/chunk ownership、计算负载、KV/O 通信与计算通信 overlap |
| 直接减少的浪费 | 短样本使用过大 CP degree 导致的冗余切分、通信和 rank 空转 | 同一 CP group 内因 mask/layout 导致的负载不均、冗余 KV/O 传输和 overlap bubble |
| 不负责 | MLA projection、DSA indexer/top-k；也不自动改变组内 attention 算法 | 跨样本选择 CP degree、全局 DP×CP sample scheduling；也不拥有 MLA/DSA 模型参数 |

当前 pinned Megatron 的实际执行是：scheduler 根据 `ceil(seq_len / max_seq_len_per_rank)` 选择向上取整到 2 的幂的 CP size，写入 `local_cp_size`；batch slicing 再绑定对应 hybrid DP×CP process group，并在该 group 内继续使用 per-sequence balancing。它是 sample/group scheduler，不是 DSA sparse executor。

当前能力边界必须如实记录：

1. Megatron 普通 `MLASelfAttention` 在收到 `packed_seq_params.local_cp_size` 时显式报错，说明 hybrid CP 尚不支持 MLA。
2. `AbsorbedMLASelfAttention` 同样显式报错，DSA 又建立在 absorbed MLA 上；因此当前 dynamic CP 不能直接作为 MLA+DSA baseline。
3. 当前 Megatron DSA 的 CP 路径只允许 `cp_comm_type=allgather`，会聚合 latent key/value 与 indexer key 后再做 global top-k。dynamic CP 即使未来接通，也主要通过缩小 group 降低成本，不会自动把组内 AllGather 变为 top-k-directed sparse communication。
4. Magi PR #331 只补齐 expanded attention 的 `Q/K=192,V=128` 和 `256/256` FA4/communication ABI，不包含 MLA projection、absorbed MLA 或 DSA indexer。
5. Magi direct IndexAttn 已有单 GPU token-index kernel，但当前 distributed executor 仍调用 `sparse_load=False,index_attn=False`，测试明确标注 `No distributed sparse yet`。所以“分布式 exact-top-k + sparse latent-KV communication”是本项目要新增并证明的能力，不是现有 Magi 功能。

推荐的最终组合层次是：

```text
Variable-length global batch
  -> Dynamic CP scheduler（可选外层；选择 sample -> CP subgroup）
  -> CPLayoutProvider（在选中的 subgroup 内建立 physical layout）
  -> Megatron MLA/DSA semantic preparation
  -> Native TE/DSA 或 Magi distributed execution
  -> local loss / backward
```

但实施顺序不能直接从组合系统开始：

1. **固定 CP 先证 Magi 组内价值**：Megatron native fixed CP 对 Magi fixed CP，避免 scheduler 差异掩盖 execution 差异。
2. **单独证 dynamic CP 调度价值**：使用相同 native backend、真实 variable-length distribution，比 static CP 与 dynamic CP；在 MLA 尚不支持时只作为标准 attention control，不冒充 MLA 结果。
3. **MLA+DSA 先做固定 CP**：native AllGather oracle、Magi conservative AllGather、Magi sparse communication 三者逐级比较。
4. **最后组合 dynamic CP + Magi**：只有前两条各自通过，才让 scheduler 选择 subgroup，再在 subgroup 内使用 Magi；组合结果不能替代两项独立证据。

Dynamic CP 与 Magi 的价值矩阵如下：

| Workload | Dynamic CP 预期价值 | Magi 预期价值 | 首轮结论 |
| --- | --- | --- | --- |
| 长度差异大、组内为规则 dense causal | 高：避免短样本过度 CP | 未知，可能被 planning/dispatch 开销抵消 | dynamic CP 是强 baseline，Magi 必须实测 |
| 长度接近、但 mask/chunk 工作量不均 | 较低 | 高：组内 layout/load balance/communication overlap | 重点验证 Magi |
| MLA dense causal、单节点 CP<=4 | 中等但当前 Megatron 未接通 | 未知；PR #331 仅提供执行 ABI | 不能预设 Magi 会赢 |
| MLA+DSA exact top-k | 只能选择 group size，不能独自解决组内稀疏路由 | 潜在核心价值，但 distributed sparse 仍需实现 | 本项目主要研究假设 |

### 1.2 Automodel #2384 的启发与边界

参考：[NVIDIA-NeMo/Automodel#2384](https://github.com/NVIDIA-NeMo/Automodel/pull/2384)。它最有价值的不是证明 Magi 比某个 backend 快，而是提供了一条已经进入真实训练 recipe 的生命周期证据：

```text
MagiState/setup_magi（进程/recipe 级）
  -> prepare_llm_batch（每 step/microbatch）
  -> packing + logical/physical sequence metadata
  -> build Magi runtime key
  -> dispatch input_ids + labels + position_ids
  -> registered Magi attention
  -> local token loss
  -> CP-aware loss numerator / valid-token count reduction
```

本项目吸收以下设计：

1. **增加薄的长期 integration facade**：使用 `MagiIntegration` 管理可选依赖、静态能力、process-group binding 和 adapter factory，让训练 recipe/benchmark runner 保持很薄；每个 microbatch 的动态状态仍放在显式 `CPRuntimeContext` 中，二者不能合并。
2. **E2E benchmark 从 batch materialization 开始**：headline 必须计入 batch 生成、THD packing、key/plan build、dispatch、attention、local loss 和 backward。只测 `calc_attn(Q,K,V)` 只能作为 breakdown，不能替代黄金 benchmark。
3. **区分逻辑与物理 packed metadata**：schema 同时保存 logical `cu_seqlens` 与 physical/padded `cu_seqlens_padded`；Magi dispatch、position IDs 和 local tensor 长度使用 physical metadata，document legality、有效 token 统计和 DSA selection 使用 logical boundary + valid-token mask。
4. **默认 local loss**：labels 与 loss mask 使用同一 layout dispatch，logits 保持 local；只归约 loss numerator 和 valid-token count，不 undispatch `[tokens,vocab]` logits。任何需要完整 logits 的 fused loss 必须在 distributed init 前明确拒绝或提供单独实现。
5. **Q/K/V layout 是 correctness contract**：进入 Magi 前显式验证 shape、stride、dtype 和 contiguous；为 rearrange 后 non-contiguous backward/NaN 增加回归测试。第一版 BF16 是明确 capability，不允许 backend 内静默改 dtype。
6. **capability gate 前置**：representation、QK/V dim、head dim、dtype、packing、CP、loss implementation 等不支持组合，必须在初始化 distributed/process groups 前失败。
7. **最终序列确定后才能 dispatch**：Automodel 的 VLM 限制说明，如果 token merge/packing 会改变最终序列，layout planning 不能提前。本项目同理：Hybrid 第一版采用 attention-local dispatch/combine；persistent layout 必须先解决 KDA/GDN 有序 recurrent state。

以下实现方式不能照搬：

- Automodel 路径是普通 expanded QKV，并显式限制 `QK dim == V dim`、`head_dim <= 128`；本项目 PR #331 基线是 `Q/K=192,V=128`，而最终目标是 absorbed MLA+DSA。
- 进程全局的 active CP group、recent key 或 key cache 无法可靠覆盖 Megatron PP interleaving、activation recompute、多个 in-flight microbatch 和 dynamic CP subgroup；必须显式按 `microbatch_id + pipeline_stage + subgroup identity` 绑定 context。
- 通过模块类名写入 CP group 不属于 Megatron-native integration；本项目继续使用 `ModuleSpec` 与显式 `ProcessGroupCollection`。
- 其验证重点是收敛与功能，不包含本项目要求的同环境 E2E latency、peak memory、通信 payload、output/gradient/indexer-loss parity；黄金 benchmark 仍必须独立建立。

Dynamic CP 下的组合顺序因此进一步冻结为：

```text
dynamic CP scheduler 选择 subgroup
  -> MagiIntegration.bind(subgroup, microbatch metadata)
  -> 生成该 subgroup 专属 CPRuntimeContext/runtime key
  -> dispatch + execution + local loss
```

不能在进程初始化时创建一个固定 recent key，再让不同 local CP degree 复用。

因此，选择 Magi 的理由不是“dynamic CP 无效”，而是内部目标还包含 dynamic CP 不负责的组内问题：MLA 非对称/latent 表示、DSA exact global top-k、top-k-directed compressed latent-KV 通信、反向路由和 overlap。若固定 CP 的 dense MLA 没有收益，项目仍可继续验证独立的 DSA sparse-communication 假设；若 DSA 最终也不能在 exact parity 下同时降低 E2E、payload 和显存，则停止 Magi 主线，保留 Megatron dynamic/native CP，不以集成完成度代替项目价值。

## 2. 成功标准

项目只有同时满足以下条件才算有价值：

1. Magi 与 Megatron 原生路径使用同一 batch、同一模型参数、同一 loss 和同一并行拓扑。
2. Dense MLA 与 MLA+DSA 的 output、input gradient、全部 parameter gradients、indexer loss 和 checkpoint 语义对齐。
3. 主性能数字从 batch 获取/构造开始，包含 layout planning、dispatch、position/packed metadata、整层前反向；不能用预生成 local Q/K/V 的结果代替。
4. 训练集成数字继续包含 Embedding、LM head、local token loss 和 backward。
5. 在至少两个 ultra-long case 上，Magi 的 max-rank E2E p50 相对该 case 的**最快正确 Megatron native baseline** 提升至少 10%，且任一主 case 不允许回退超过 5%。
6. 峰值显存不得高于 baseline；MLA+DSA 还必须证明 CP 通信 payload 实际下降，而不仅是 sparse compute。
7. Megatron Core 不直接依赖 MagiAttention，不劫持 TE 全局 backend，不复制整段 attention forward，不维护长期私有 fork。
8. Dynamic CP 的 scheduler 收益与 Magi 的 group-internal execution 收益必须分别报告；禁止用“dynamic CP + Magi”的组合数字掩盖其中一项单独回退。

## 3. 黄金 Benchmark 定义

### 3.1 两个 headline harness

#### A. Attention-layer E2E

```text
BatchSource.next / BatchFactory.materialize
  -> H2D（存在时）
  -> CP layout planning
  -> Megatron zigzag/per-document slicing 或 Magi dispatch
  -> local position IDs / PackedSeqParams
  -> shared Embedding
  -> Megatron MLA semantic preparation（LoRA/projection/RoPE/indexer）
  -> Megatron native 或 Magi distributed attention execution
  -> Megatron finalization / output projection
  -> deterministic output gradient
  -> backward
```

该 harness 回答“把 Megatron attention layer 换成 Magi distributed execution 后，整层训练是否更快”。Embedding 权重固定且两端共享；输出 canonicalization 只用于 correctness，不进入 timed path，除非当前 layout policy 明确要求 combine。

这里的“真实 Megatron attention layer”必须从 hidden states 开始运行 Megatron 的 Q/KV LoRA、RoPE、MLA projection 和 output projection。Magi 路径不能从预生成 Q/K/V 开始；PR #331 只替换中间的 expanded distributed-attention execution。因此即使 Magi 不拥有 MLA 参数，本 harness 仍然是完整的模型级 MLA E2E，而不是 Q/K/V microbenchmark。

#### B. One-layer training E2E

```text
上述全部步骤
  -> shared LM head
  -> 与当前 local layout 对齐的 labels/loss_mask
  -> local cross entropy numerator + valid-token count
  -> CP reduction
  -> loss backward
```

该 harness 回答“进入真实训练链路后收益是否仍然存在”。默认不 undispatch `[tokens, vocab]` logits。

### 3.2 四组正式比较

| 训练阶段 | Megatron oracle / baseline | Magi 路径 | 要回答的问题 |
| --- | --- | --- | --- |
| Dense MLA | 真实 Megatron MLA layer；对每个 case 取最快 native CP mode | Megatron MLA semantic prep/finalize + PR #331 `expanded_qkv+dense` execution | Magi 的 layout/dispatch/communication/kernel 在完整 layer 训练中是否有净收益 |
| MLA+DSA core | Megatron `MLA+DSA` AllGather oracle；同一 MLA/indexer 参数 | Magi exact distributed top-k + sparse latent-MLA executor | 在不改变 MLA/DSA 语义时，能否同时减少 sparse compute 与 CP communication |
| Hybrid pretraining | `GDN -> GDN -> GDN -> MLA`，canonical Megatron layout | 同一 GDN 参数与 MLA 参数；MLA 由 Magi 执行 | Magi 接入 3:1 混合线性架构后，收益是否仍覆盖 layout 转换成本 |
| Hybrid posttraining | 同一 hybrid stack，将 MLA 层升级为 Megatron `MLA+DSA` AllGather oracle | 复用已经独立通过的 Magi MLA+DSA executor | 已证明的 MLA+DSA 能力进入 3:1 stack 后，收益是否仍覆盖 layout 转换成本 |

四组比较都从 backend-neutral global batch 开始，禁止直接给两端喂预生成 local Q/K/V。MLA+DSA core 必须先独立通过，Hybrid 才能消费它；Hybrid stack 是训练集成验证，不替代两个 headline harness。

#### 3.2.1 Dynamic CP 控制组

Dynamic CP 不是新的 headline attention backend，而是额外实验轴。runner 必须能表达以下控制组，并对当前不支持的组合产出 `unsupported + reason`，不得静默换模型：

| ID | Scheduler | Group-internal backend | Architecture | 目的 |
| --- | --- | --- | --- | --- |
| `C0` | static CP | Megatron native | 标准 dense attention | 验证 variable-length E2E 基准 |
| `C1` | dynamic CP | 与 `C0` 相同的 Megatron native backend | 标准 dense attention | 只测 scheduler/group-sizing 收益 |
| `M0` | static CP | Megatron native | dense MLA | MLA 原生黄金 baseline |
| `M1` | static CP | Magi | dense MLA | 只测 Magi 组内执行收益 |
| `M2` | dynamic CP | Megatron native | dense MLA | pinned Megatron 当前应报告 unsupported；未来接通后才启用 |
| `M3` | dynamic CP | Magi | dense MLA | `C1` 与 `M1` 分别通过后才启用的组合实验 |
| `D0` | static CP | Megatron AllGather | MLA+DSA | exact 语义 oracle |
| `D1` | static CP | Magi conservative AllGather | MLA+DSA | 集成与通信接管 baseline |
| `D2` | static CP | Magi sparse latent executor | MLA+DSA | DSA 核心价值实验 |
| `D3` | dynamic CP | Magi sparse latent executor | MLA+DSA | 最终可选组合，不进入第一版 scaffold |

禁止把 `C1` 的 standard-attention dynamic CP 数字直接和 `M1/D2` 比较后宣称 Magi 更快或更慢；它只验证 harness 能正确计入 scheduler、subgroup selection 和变长 batch 生命周期。`M2/M3/D3` 在 pinned 版本没有真实支持时应保留为 capability matrix 中的明确空位。

### 3.3 双计时口径

- `batch_to_backward_wall_ms`：headline。从确定性 `BatchSource.next()` 开始，到 backward 与必要 CP loss reduction 完成；包含 synthetic batch 生成、TP broadcast、H2D、layout planning 和 dispatch。首版不包含磁盘 IO 与 tokenizer，避免把数据集系统噪声误算成 attention 性能。
- `gpu_pipeline_ms`：同一 global GPU batch 已就绪后开始，到 backward 完成；用于排除数据管线噪声。

两个总时间都必须提供以下 breakdown：

- batch materialization、TP broadcast 与 H2D；
- layout solver/key creation；
- dispatch/slicing；
- Embedding 与 metadata/RoPE；
- projection；
- distributed attention communication；
- attention kernel；
- output projection/LM head/loss；
- backward；
- dynamic CP scheduler、subgroup bind/synchronization、额外 communicator/buffer memory（启用时单列）；
- combine（如果发生）。

### 3.4 公平性不变量

- `BatchFactory` 用固定 seed 生成 tokens、labels、loss mask、position IDs、document lengths 和 padding；每个 backend 消费逐元素相同的 global batch。
- Native 与 Magi module 从同一 `state_dict` 构造；optimizer state 需要深拷贝，禁止共享 momentum tensor。
- 每个 timing sample 使用相同 batch recipe；A/B 顺序交替，防止频率、温度和 JIT 顺序偏差。
- Dense MLA 先分别验证所有 native CP mode，再把最快正确 mode 定为该 case 的 headline baseline；不得固定一个较慢 mode 来制造 Magi speedup。
- 所有 wall-time 取所有 rank 最大值；报告 median、p10、p90、min、max、样本数和 CV。
- 首轮 performance：5 次 warmup、20 次 measurement；大于 128K 的 case 使用 3 次 warmup、10 次 measurement。CV 超过 5% 时重跑一次，仍超标则标记不稳定，不形成性能结论。
- 第一次运行的 JIT/compile 时间单独记录，不混入 steady-state 数字。
- Correctness 所需的 canonicalization 可以放在 timed region 之外；但如果实际训练路径为了进入下一层必须 combine/redispatch，该转换必须留在 timed region。

### 3.5 结果 schema

每个 JSON 结果必须保存：

- case 全配置和 schema version；
- attention representation、selection、claim scope 和 backend capability snapshot；
- Magi、Megatron、PR #331、flash-attention submodule SHA；
- Git branch、dirty status 和 patch hash；
- Python、Torch、CUDA、TE、CuTeDSL、NCCL 版本；
- GPU UUID/name/capability、拓扑、时钟、温度和关键环境变量；
- correctness 指标；
- 两个 headline latency 和 breakdown；
- logical/physical tokens/s；
- max-rank allocated/reserved memory；
- collective/P2P payload bytes 与调用次数；
- nsys/ncu artifact 路径。

## 4. 统一 case 与 adapter 架构

### 4.0 `schema`、`batch`、`adapters`、`runner` 分别是什么

可以把一次 benchmark 想成一场标准化实验：

| 组件 | 白话含义 | 它负责什么 | 它不负责什么 |
| --- | --- | --- | --- |
| `schema` | 实验订单/规格书 | 声明要测什么：MLA 还是 MLA+DSA、序列与 varlen 形状、TP/CP/PP、backend、harness、正确性阈值和计时次数 | 不生成 tensor，不执行模型，不包含某个 backend 的 Python 对象 |
| `batch` | 两个选手共用的标准试样 | 根据 schema 和 seed 生成逐元素相同的 tokens、labels、loss mask、position、document/global token IDs、logical/padded `cu_seqlens` | 不决定使用 Megatron zigzag 还是 Magi layout，不运行 attention |
| `adapters` | 两套后端转接头 | 把同一个 schema 和 global batch 翻译成该后端的真实执行：Native adapter 走 Megatron slicing/ModuleSpec；Magi adapter 走 planning/dispatch/`calc_attn` 或 MLA+DSA executor | 不各自维护一套 benchmark 主循环，不偷偷更换 batch、参数、loss 或计时口径 |
| `runner` | 唯一裁判和实验流水线 | 读取 schema、调用 BatchFactory、复制同一 state dict、依次驱动 adapters、做 correctness、warmup、计时、max-rank 聚合并写 JSON | 不实现 Megatron/Magi 算法，不在主循环里复制两套 backend-specific forward |

其中 `schema` 还分两面：输入 case schema 规定“要跑什么”，输出 result schema 规定“结果必须怎么保存”。本文通常单独说 `schema` 时，优先指输入的 `AttentionBenchmarkCase`。

完整关系是：

```text
case schema
    -> runner
       -> BatchFactory -> immutable GlobalBatch
       -> shared state_dict
       -> MegatronNativeAdapter -> Native result
       -> MegatronMagiAdapter   -> Magi result
       -> correctness + timing + memory + communication report
```

这四个组件拆开，是为了确保后续中等推理强度模型只需按任务卡实现小模块：增加 case 时不改 runner，增加 backend 时只加 adapter，修 batch/varlen 时两端自动同时受益。

### 4.1 两个正交维度：representation × selection

MLA/DSA 能力不能只用一个 `attention_mode` 模糊表示，必须同时记录执行表示与选择方式：

| 执行表示 | Dense selection | Exact top-k selection |
| --- | --- | --- |
| `expanded_qkv` | **第一条真实 Magi golden**：Megatron 完成 MLA projection/RoPE，Magi PR #331 接收 `Q/K=192, V=128` 并做 dense distributed attention | 允许作为 correctness/成本诊断，但会物化或通信 expanded K/V，不能作为最终 MLA+DSA 价值路径 |
| `absorbed_latent` | Megatron absorbed MLA reference/fallback；用于验证 latent 表示、V-up ownership 和 pre-AllGather seam | **最终目标**：Megatron 准备 absorbed query、compressed latent KV 与 indexer tensors，Magi 完成 exact distributed top-k、稀疏通信、attention 与 backward |

这里的 `expanded`/`absorbed` 描述执行时的张量表示，不改变模型仍然是 MLA：

- `expanded_qkv`：MLA projection 后显式形成 per-head Q/K/V；PR #331 解决的是 Magi runtime/kernel 对非对称 K/V head dim 的支持。
- `absorbed_latent`：将可吸收的 projection 代数并入 query/V-up 路径，通信和 attention 直接围绕 compressed latent KV，不显式展开完整 per-head K/V。
- `dense`/`exact_topk`：分别表示所有合法 KV 参与 attention，或由 DSA indexer 选择 exact global top-k；DSA 是 MLA 上的选择与执行策略，不是另一套模型参数化。

首轮正式路径只承诺两个对角目标：`expanded_qkv+dense` 建立 PR #331 与 Megatron 的黄金 E2E，`absorbed_latent+exact_topk` 建立最终 MLA+DSA。其他组合必须显式标记 `diagnostic/reference`，不能进入 headline value claim。

### 4.2 `AttentionBenchmarkCase`

统一配置必须表达：

- `architecture`: `mla_only | hybrid_linear`；
- `attention_mode`: `dense_mla | mla_dsa`；
- `attention_representation`: `expanded_qkv | absorbed_latent`；
- `claim_scope`: `headline | diagnostic | reference`；
- `linear_backend`: `none | megatron_gdn | internal_kda`；
- 模型维度：hidden size、Q heads、KV heads、Q/K/V/latent dims、LoRA ranks、DSA indexer/top-k；
- 序列：SBHD/THD、global tokens、logical/padded document lengths、microbatch size；
- 并行：TP/CP/PP/DP sizes 与 process groups；
- runtime：dtype、kernel/backend、warmup、iterations、seed、loss policy；
- layout：`megatron_zigzag | magi_attention_local | magi_persistent`。

该 schema 不 import Megatron 或 Magi，必须能在无 GPU 环境中 dry-run 和校验。

Validation 冻结以下规则：

- `dense_mla + expanded_qkv` 是 PR #331 主路径；
- `mla_dsa + absorbed_latent` 是正式 DSA 路径；
- `mla_dsa + expanded_qkv` 只能设置 `claim_scope=diagnostic`；
- backend 必须在 distributed 初始化前声明是否支持该 representation、selection、packed、backward 和 CP size，不能运行到 kernel 才静默 fallback。

### 4.3 `BatchFactory`

输出一个 backend-neutral `GlobalBatch`：

```text
tokens, labels, loss_mask, position_ids,
logical/padded cu_seqlens, document IDs,
valid-token mask, global token IDs, seed/provenance
```

Packed-varlen 不能依赖“每个 document 长度可被 2*CP 整除”。必须支持 logical/padded lengths 不同，并提供 oracle 验证任何 query 都不会跨 document 选择 KV。

### 4.4 `MegatronNativeAdapter`

- 通过 pinned Megatron 的公开 `ModuleSpec` 构造真实 `MLASelfAttention` 或 `AbsorbedMLASelfAttention + DSAttention`；后者在文档和结果中统一标记为 `MLA+DSA`。
- Dense MLA 使用 Megatron 原生 batch CP slicing、per-document packed layout、RoPE 与实际支持的 CP modes；`MLA+DSA` 使用其当前 AllGather CP 与可用 native DSA kernel backend。
- 同一 builder 必须暴露“semantic preparation、attention execution、finalization”三个可测边界；Native 路径把三者原样组合，不能只导出 Q/K/V tensor 作为所谓 attention layer。
- 不复制 Megatron attention 实现；该 adapter 是 correctness oracle 和 baseline。

### 4.5 `MegatronMagiAdapter`

- 复用同一 Megatron Embedding、MLA projection、RoPE、参数、output projection、indexer 和 loss；它们全部留在真实 autograd graph 内。
- 组合 `CPLayoutProvider` 与 `AttentionExecutionBackend`，只替换 token layout 和 distributed attention execution，不重新实现完整 MLA forward。
- Dense MLA 由 Megatron 产生 expanded Q/K/V，再调用 PR #331 增强后的 Magi runtime，显式传 `head_dim=192`、`head_dim_v=128`。
- `MLA+DSA` 阶段由 Megatron/model 拥有 indexer 参数、scoring/top-k 语义、indexer loss、index-share 与 checkpoint；Magi 接收 local indexer representations 和 exact top-k contract，执行 distributed exact top-k、latent-KV/Q routing、sparse MLA kernel 与 backward。
- Magi 只搬运 compressed latent KV 与必要 positional component，禁止先展开为 per-head dense K/V 再通信。
- `up_v_weight` 始终是 Megatron 参数；Magi 可以接收 borrowed weight tensor 并在 kernel 中融合 V-up，但不得注册副本。Result 用 `v_up_applied` 明确告诉 Megatron是否还需 finalization。
- Adapter 返回原 query owner 对应的 local output；是否 combine 由 layout policy 决定，indexer loss/share 和 output projection 仍由 Megatron finalizer 完成。

代码归属：

- Megatron semantic adapter 与 typed contract：`magi_attention/integrations/megatron/`；
- 通用 distributed top-k、routing、transport 与 sparse execution：`magi_attention/sparse/`，不得 import Megatron；
- benchmark runner/cases：`exps/megatron_attention/`；
- focused tests：`tests/test_megatron_integration/`。

## 5. Megatron 优雅集成边界

### 5.1 三层职责与完整 E2E 数据流

正式架构固定为三层，不让 MagiAttention 变成第二套模型框架：

```text
GlobalBatch / canonical hidden states
  -> CPLayoutProvider
       决定 physical token layout、local positions 与 packed ownership
  -> Megatron semantic preparation
       Embedding、Q/KV LoRA、norm、RoPE、expanded 或 absorbed MLA 表示
       DSA 模式额外产生 local indexer Q/K/weights 与 exact-top-k contract
  -> AttentionExecutionRequest（typed、不拥有参数、带 autograd tensors）
  -> AttentionExecutionBackend
       Native Megatron 或 Magi distributed execution
  -> AttentionExecutionResult
       local output、top-k/aux、V-up 状态、通信统计
  -> Megatron finalization
       可选 V-up、indexer loss/share、output projection、residual、LM head/loss
```

因此“PR #331 没有完整 MLA projection”不是架构缺口：完整 MLA 流程由 Megatron semantic preparation/finalization 提供，PR #331 只实现 `expanded_qkv+dense` execution backend。Benchmark 仍从 batch 生成开始并穿过整个组合，不会把 Q/K/V 当输入起点。

参数和状态所有权冻结如下：

| 对象 | Owner | Magi 可做的事 | 禁止事项 |
| --- | --- | --- | --- |
| Q/KV LoRA、norm、RoPE、output projection | Megatron ModuleSpec | 消费这些模块产生的 tensors | 在 Magi 内复制或重新注册参数 |
| DSA indexer 参数、scoring 语义、loss、index-share | Megatron | 对带 autograd 的 local indexer representations 做 distributed exact top-k | 改变 scoring/tie-break，或在 backend 保存第二份 indexer state |
| `up_v_weight` | Megatron | 以 borrowed tensor 融合进 absorbed kernel | 改变 state-dict key、parameter identity 或 optimizer ownership |
| TP/PP/CP groups、checkpoint、recompute 生命周期 | Megatron | 使用显式传入的 process groups 和 microbatch identity | 读取隐式 global group、跨 PP process 传 Python key |
| layout plan、communication schedule、temporary buffers | Magi | 创建非持久 runtime state并记录通信统计 | 将 runtime cache 写入模型 checkpoint |

`AttentionExecutionBackend` 第一版必须是 parameter-free `nn.Module` 或普通 protocol implementation。未来如果算法确实需要新 learnable state，必须先升级模型 schema、Native oracle 和 checkpoint contract，不能暗中塞入 Magi backend。

### 5.2 Dense MLA：layout/runtime abstraction

Magi 需要两个对象，而不是让裸 `DistAttnRuntimeKey` 污染所有 MCore API：

```text
CPLayoutProvider（模型/进程级）
  -> prepare(batch metadata, local attention metadata, process groups)
CPRuntimeContext（microbatch 级）
```

`CPRuntimeContext` 持有：

- physical dispatch key；
- per-mask attention keys；
- local/global token mapping 和 local position IDs；
- logical/padded packed metadata；
- TP-local Q/KV head metadata；
- PP-stage-local CP group；
- local/global loss policy；
- microbatch/virtual-pipeline identity。

Megatron 侧只暴露 backend-neutral context 和 `ModuleSpec` 插槽，不 import Magi 类型。Magi adapter 解包 backend state 后调用 `dispatch`/`calc_attn`。

该方向借鉴 NVIDIA/Megatron-LM PR #2749 的 CP handler 生命周期，但不直接依赖该 WIP PR：正式接口继续拆分长生命周期 provider 与 microbatch runtime context，避免一个 handler 同时承担 backend strategy、packed metadata、layout state 和所有模型操作。

`CPLayoutProvider` 与 execution backend 是两个独立扩展点：前者在 batch/hidden-state 边界决定 token 在哪里，后者接收已经对齐该 layout 的 semantic tensors。DSA backend 可以在内部进一步路由 compressed KV 或 query，但不能悄悄改变 caller-visible token ownership；返回前必须恢复 request 中的 query-owner 顺序。

### 5.3 PP 与 TP

- TP：runtime key 使用实际 TP-local Q/KV head 数，不能使用 global config head 数。
- PP：stage 0 只 dispatch tokens 一次；CP-local activation 按相同 local CP coordinate 穿过后续 stages。
- 每个 PP stage 根据共享的可序列化 layout plan 和本 stage 的 `pg_collection.cp` 创建自己的 runtime key；禁止跨进程传 Python key。
- Context 绑定 microbatch 和 virtual stage，支持 interleaving 与 recompute。

### 5.4 Loss 与 RoPE

- RoPE 使用 context 提供的显式 global position IDs，不再假设 local positions 连续。
- labels 与 loss mask 使用同一 dispatch key；默认在最后 PP stage 本地计算 loss numerator/token count，再在 CP/DP 域归约。
- 只有调试/推理确实需要 global output 时才调用 undispatch。

### 5.5 MLA+DSA：pre-AllGather execution seam

普通 core-attention backend 位于 dense gather 之后，无法减少 DSA CP 通信。MLA+DSA 需要一个 typed pre-AllGather request/response seam：

```text
Megatron semantic preparation
  -> local absorbed MLA tensors
  -> local indexer Q/K/weights + exact scoring/top-k contract
  -> packed/global ownership metadata + separate MLA/indexer RoPE semantics
  -> Magi distributed exact top-k
  -> Magi sparse latent-MLA communication/execution
  -> output + top-k/result metadata
  -> Megatron finalizes indexer loss/share state
```

Megatron 继续拥有模型参数、indexer loss、index-share、checkpoint 和 fallback；Magi 不复制 `DSAttention.forward`。Stable request 中不携带 Megatron fallback callable：backend 返回 `UnsupportedExecution`/`None` 后由 Megatron adapter 调原生路径，避免 Magi 反向依赖 Megatron 私有实现。

当前 Magi 的 direct `IndexAttn` 是单 GPU sparse kernel，仓库测试仍明确写着 `No distributed sparse yet`。因此 distributed MLA+DSA 是本项目要实现和证明的能力，不能把现有单 GPU adapter 当作已完成的分布式方案。

## 6. 执行阶段与验收门

### 6.0 已知卡点与降级边界

整体骨架可以直接实现，但以下项目不是普通 glue code，必须用对应 Gate 先证伪：

| 卡点 | 最早验证任务 | 失败时允许的降级 | 会阻塞什么 |
| --- | --- | --- | --- |
| PR #331/FA4 在当前 SM100、Torch、CuTeDSL、submodule 组合下的 192/128 backward | T01–T02 | 保留 patch provenance，Native runner继续完成；不得用不同环境比较 | 阻塞 Magi expanded dense performance，不阻塞 schema/native scaffold |
| Megatron expanded MLA tensor layout 与 Magi `Q/K=192,V=128` ABI 是否逐维一致 | T06–T08 | 只在 adapter 做显式、计时内的 view/transpose；不改数学语义 | 阻塞 Gate 2，需先冻结 shape contract |
| absorbed MLA 的 latent layout、positional component、V-up 与 DSA indexer 双 RoPE 语义 | T11 | 使用 Native absorbed AllGather reference并标记 takeover=false | 阻塞 Magi MLA+DSA execution，不阻塞 Native oracle |
| Magi 当前无 distributed exact sparse；top-k 与反向路由都是新能力 | T12–T13 | compressed latent KV AllGather + local sparse reference | 阻塞 Gate 3B value，不阻塞 Gate 3A scaffold |
| PP middle stage 没有原始 batch，但需要同一 physical layout 的 stage-local key | T10 | 第一版限制 PP=1 只能作为 partial evidence，不能宣布集成完成 | 阻塞“优雅集成”与 Hybrid PP gate |
| packed padding、document boundary 与 non-contiguous owner mapping | T04/T08/T12 | 先缩小到已记录的 supported fixture；不得删除 padding 后宣称 varlen 支持 | 阻塞对应 THD headline case |
| KDA/GDN recurrent state 依赖全局 token 顺序 | T14 | 永久保留 canonical storage + attention-local dispatch/combine | 只阻塞 persistent layout，不阻塞 MLA+DSA 主线 |

其中真正的研究卡点是 absorbed sparse execution、distributed exact top-k/gradient routing 与 KDA persistent layout；其余属于可通过 contract、adapter 和环境固定解决的工程卡点。任何降级都必须进入 result `limitations`，不能把 reference/fallback 数字混入 Magi value claim。

### Phase 0：来源、分支和环境冻结

1. 记录已经对齐的 `magic` 分支状态；确认方案前不继续 push、commit 或启动同步。
2. 执行获批后，先在临时 clean worktree fetch PR #331，检查完整 diff、submodule pin、与当前用户 dirty paths 的交集和对 `magic` 的可重放性；#326 只作测试/失败案例参考，不应用。
3. 只有 clean-worktree preflight 通过后才把 #331 以独立 provenance commit 合入 `magic`；每次只 stage 本项目文件，并在前后核对用户文件哈希。
4. 创建独立远端环境 `/root/work/envs/magi-megatron-bench`。候选基座是已验证的 Python 3.12 环境，但最终 Torch/CUDA/TE/CuTeDSL/FA4 版本由 PR #331 与 pinned Megatron 的共同兼容探针决定；两端必须使用同一个 Python executable，禁止用不同 Torch 栈形成伪比较。
5. 在同一环境运行 PR #331 的 SM100 192/128 forward/backward、varlen 和 CP1/2/4 focused tests，并运行 Megatron MLA/MLA+DSA import 与单层 smoke。

Gate 0：两端 Git/patch/submodule SHA 对齐；用户文件哈希未变；同一 Python 进程能 import Megatron、TE、Magi；192/128 CP4 前反向通过；保存完整环境 manifest。远端 GPU 名称、UUID、SM、显存与拓扑在这里重新取证，不仅依赖历史探针。

### Phase 1：Native Megatron 黄金基线

1. 将 LongAttention 中可复用的 ModuleSpec runner 与 provenance schema 迁移进 Magi repo。
2. 新建 backend-neutral case、BatchFactory 和 result schema。
3. 跑通 MLA CP1/2/4 的 SBHD 与 packed THD；对每个 case 枚举所有实际支持的 native CP modes 与 kernel backends，加入 attention-layer 与 one-layer-training 两个 harness。
4. 保存 output/gradient oracle 和 NSYS NVTX 范围。

Gate 1：所有 native case 可复现；两次独立运行数值一致；结果包含完整 provenance；无 Magi import。

### Phase 2：Magi expanded dense MLA 与公平 E2E 对照

1. 实现 `CPLayoutProvider`、`CPRuntimeContext` 和 `ExpandedMLAExecutionBackend` adapter；该 backend 接收真实 Megatron MLA projection 产生的 expanded Q/K/V。
2. 从同一 global batch 开始，分别运行 Megatron native slicing 与 Magi planning/dispatch。
3. 对齐 output、input/parameter gradients 和 local loss；correctness 通过后才启用 timing。
4. 运行公开矩阵并保存 nsys 报告；对瓶颈 kernel 再使用 ncu。

Gate 2：正确性通过，结果能明确拆分 planning、dispatch、comm、kernel 与 backward，并给出相对最快 native baseline 的可信结论。Dense MLA 若未达到 10% speedup，不宣称 dense backend 有价值，但仍可进入 MLA+DSA，因为 DSA 的 sparse communication 是独立价值假设。

### Phase 3：Megatron 集成、保守 absorbed MLA+DSA 骨架与 distributed sparse

1. 先用 expanded dense MLA 冻结 backend-neutral `CPLayoutProvider/CPRuntimeContext`、ModuleSpec 注入、TP-local head metadata 和 PP stage-local runtime lifecycle；这是 absorbed MLA+DSA 能优雅进入 Megatron 的前置接口，不依赖 Hybrid stack。
2. 建立 Megatron native oracle：同一 MLA/indexer 参数，AllGather latent KV 与 indexer K，得到 exact global top-k、indexer loss 和 sparse MLA output。
3. 先实现保守 `MagiDSAReferenceBackend`：保持 Megatron absorbed MLA+DSA 的 AllGather、global reorder、exact top-k、indexer loss/index-share 和 sparse-attention 语义。首版允许显式调用 native fallback，JSON 必须标记 `communication_takeover=false`；随后可以由 Magi adapter 接管同等 AllGather orchestration，但仍不得宣称通信优化。

Gate 3A（DSA Scaffold）：Megatron integration seam 通过 TP/CP/PP/recompute/checkpoint；provider on/off 的 output、top-k、indexer loss、input/parameter gradients、state dict 与 optimizer identity 对齐。该 Gate 不要求 speedup，也不要求通信量下降。

4. Gate 3A 通过后，再将优化问题拆成两块。Indexer：local indexer Q 固定，stream/ring indexer K，tile scoring + running stable TopK，不物化完整 score matrix。Post-top-k executor：先保留 compressed latent KV AllGather 语义基线，再评估 Magi 可承载的 latent sparse kernel。
5. 捕获 per-query selected owners、unique selected latent-KV coverage、请求重复率、rank fanout、metadata 与反向 payload，先用数据判断通信策略。
6. 实现并公平比较两种 exact executor：selected latent-KV pull 与 query push/remote partial MLA；禁止把 latent KV 展开成 dense per-head K/V 后通信。
7. 选择真实 E2E 更快的策略；高覆盖率/高 fanout case 自动 fallback 到 compressed AllGather/native path。
8. 验证 packed document boundary、top-k global IDs、stable tie-break、indexer loss、backward gradient return、index-share，以及 indexer RoPE 与 MLA RoPE 各自的 layout 语义。

Gate 3B（DSA Value）：top-k indices（含稳定 tie-break）与 Megatron oracle 一致；MLA+DSA output/gradient/loss 通过；至少两个 ultra-long case 相对 native MLA+DSA E2E 提升 10%，通信 payload 和峰值显存同时下降。

### Phase 4：GDN+MLA/DSA 3:1 Hybrid 代理

1. 在已经通过 Gate 3B 的 dense MLA 与 MLA+DSA executor 上，建立四层 Hybrid unit：`GDN -> GDN -> GDN -> MLA`；pretraining 使用 dense MLA，posttraining 将同一个 MLA slot 升级为 MLA+DSA。多 unit 只用于扩展训练集成覆盖，不改变 3:1 pattern。
2. v1 使用 canonical Megatron layout；每个 MLA/MLA+DSA slot 执行 attention-local dispatch/combine，所有转换进入 E2E 计时。
3. LinearProxy 只验证 norm/residual/forward/backward、50-step、recompute、TP2×CP2、PP2×CP2 和 distributed checkpoint；本阶段不优化 GDN/Linear kernel。
4. 在进入 persistent layout 前先验证 GDN 的有序 recurrence 不变量。依次评估：KDA-friendly fixed/order-preserving layout；显式跨 rank recurrent-state handoff；最后才是一般 Magi non-contiguous layout。禁止仅把 context 传给 GDN 就声称支持。
5. 只有某个方案同时保持全局 token 顺序语义与 state propagation 正确，才构造 v2 persistent layout，让整个 hybrid block 只做一次物理 dispatch。

Gate 4：attention-local v1 的 pretraining/posttraining 都必须正确；v2 只有在 recurrent state/output/gradient 全部对齐且相对 v1 E2E 提升至少 5% 时才保留。否则正式方案维持 canonical storage + attention-local dispatch/combine，不为“少一次重排”牺牲 KDA 语义。

### Phase 5：替换为内部 KDA 与真实训练配置

1. 将 GDN adapter 换成内部 KDA ModuleSpec，不改变 benchmark/CP runtime contract；重新验证 KDA recurrence 与选择的 storage/layout policy。
2. 保持已冻结的 3:1 KDA:MLA 层比例，导入真实 MLA dimensions、MLA+DSA top-k、packed length distribution 和 TP/CP/PP 配置。
3. 重跑 correctness、50-step training、checkpoint 和公开/内部双矩阵。

Gate 5：内部主 case 的 E2E 几何平均提升至少 10%，无 case 回退超过 5%，且能以小型 upstream-neutral Megatron patch 集成。

## 7. 第一版公开矩阵

模型采用 DeepSeek-like MLA：

```text
hidden_size=7168
num_attention_heads=128
q_lora_rank=1536
kv_lora_rank=512
qk_head_dim=128
qk_pos_emb_head_dim=64
v_head_dim=128
DSA indexer heads=64, head_dim=128, topk=2048
```

矩阵：

- Correctness smoke：4K/8K/16K，CP1/2/4。
- Dense MLA performance：16K、32K、64K、128K，CP1/2/4 中可运行的组合。
- Packed-varlen：总 physical tokens 64K 和 128K；包含 logical/padded 长度不相等、大量短文档和 boundary-heavy case。
- MLA+DSA：128K、256K，并以 1M 作为 feasibility case；1M 未完成前不得声称支持百万 token 训练。
- Parallel integration：TP2×CP2、PP2×CP2（受 4 GPU 限制分别运行）。
- Headline：BF16、forward+backward、max-rank latency；forward-only 只作诊断，不形成训练结论。

## 8. Correctness 门槛

- Dense MLA output/input-grad/param-grad：relative L2 <= `5e-3` 且 cosine >= `0.999`；同时记录 max-abs，不以单一 max-abs 决策。
- CP1 参考路径额外与 FP32/SDPA oracle 比较。
- MLA+DSA top-k global indices 必须 exact；相同分数按 global token ID 稳定 tie-break。
- MLA+DSA indexer loss relative error <= `1e-5`；禁止跨 packed document 选 KV。
- Indexer RoPE 与 MLA RoPE 分别遵循 Megatron/DeepSeek 语义，不能因为共享 `position_ids` 就共享错误的 interleaving 解释；该项需要独立 oracle。
- State-dict keys、parameter shapes 和 optimizer parameter identity 必须一致。
- 四层训练：50 steps；loss、参数更新轨迹在冻结 envelope 内；step 25 做 model/optimizer distributed checkpoint save/resume。

任何 correctness 失败都会阻止对应 performance case；不允许通过放宽阈值掩盖系统性偏差。

## 9. 交付物

- 可独立运行的 Megatron attention benchmark runner 和 JSON schema；
- Native/Magi 两套 adapter 与 correctness suite；
- 公开 case 配置和一键 CP1/2/4 matrix；
- 端到端报告：latency、throughput、memory、communication、nsys/ncu；
- GDN hybrid proxy 及 attention-local/persistent layout 对照；
- MLA+DSA exact indexer baseline、需求分布报告和最终 sparse communication strategy；
- Megatron 最小 patch series、兼容矩阵和集成文档；
- 内部 KDA/config 到位后的最终 value report。

## 10. 明确不做

- 第一版不覆盖任意 mask、SWA/full 混合 mask、FP8/FP4、CUDA Graph、inference/decode。
- 不把 #326 和 #331 同时应用。
- 不在 benchmark 证明前将 Magi non-contiguous layout 强行贯穿 KDA/GDN。
- 不把单 GPU IndexAttn、只减少 sparse compute 或只通过 microbenchmark 当成 distributed MLA+DSA 完成。
- 不把 GDN 性能称为 KDA 性能。
- 不用 core-kernel microbenchmark 冒充端到端价值。
- 不修改或清理现有用户 dirty 文件；不维护 Megatron 长期 fork。

## 11. 参考与确认点

- [MagiAttention PR #331](https://github.com/SandAI-org/MagiAttention/pull/331)：当前 MLA 192/128 与 256/256 runtime/kernel patch；明确 supersede #326。
- [MagiAttention PR #326](https://github.com/SandAI-org/MagiAttention/pull/326)：历史 WIP，仅用于理解测试范围与失败模式。
- [DeepSeek-V3.2-Exp](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp)：MLA 基础上加入 DSA 的语义参考，尤其注意 indexer 与 MLA RoPE layout 不同。
- [Megatron-LM PR #2749](https://github.com/NVIDIA/Megatron-LM/pull/2749)：CP handler 抽象方向参考；本方案不依赖其合并状态。

用户确认本方案后，才从 Phase 0 开始逐 Gate 执行；任何 Gate 失败先提交证据与决策，不自动进入下一阶段。

## 12. 第一核心里程碑：Scaffold Gate

第一轮实现只搭整体架子，名称为 **Scaffold Gate（S0）**。S0 完成不代表性能有提升，也不代表 distributed MLA+DSA 已实现；它只证明后续工作可以在一套稳定接口和同一个 runner 中逐步替换，而不需要推翻工程结构。

S0 必须同时满足：

1. `AttentionBenchmarkCase` 能表达 `mla_only`、3:1 `hybrid_linear`、`dense_mla`、`mla_dsa` 以及 `expanded_qkv | absorbed_latent`，并能在无 GPU 模式做 schema/capability validation。
2. `BatchFactory` 从固定 seed 生成同一份 global SBHD/THD batch，支持 logical/padded lengths 不同，并带 global token/document ownership oracle。
3. Runner 有且只有两个 headline mode：`attention_stack` 与 `attention_to_loss`；两者都从 batch 生成开始。
4. `MegatronNativeAdapter` 能通过真实 `ModuleSpec` 构造 dense MLA 和 native MLA+DSA；不复制 Megatron forward。
5. `MegatronMagiAdapter` 的 tagged request/result、parameter ownership、fallback 和 result schema 完成；Magi backend parameter count 为 0。S0 允许内部仍调用 native/reference path，但 JSON 必须明确写 `communication_takeover=false`，禁止伪装成 Magi 优化完成。
6. 3:1 Hybrid recipe 能构造 `GDN, GDN, GDN, MLA`，执行至少一次 CP1 forward/backward，并能通过配置将最后一层切换为 MLA+DSA。
7. 每个 run 都产出 schema-versioned JSON、环境/source provenance、correctness 状态和各阶段 timing 字段；尚未实现的字段写 `null + reason`，禁止填 0。
8. 所有用户 dirty 文件哈希不变；Magi、Megatron、LongAttention 的代码归属没有混淆。

S0 明确不做：Linear/GDN kernel 优化、persistent layout、distributed top-k、sparse KV routing、完整性能 sweep、FP8/FP4、真实内部 KDA。

## 13. 目标目录结构与代码归属

```text
MagiAttention/
├── magi_attention/
│   ├── sparse/
│   │   ├── __init__.py
│   │   ├── exact_topk.py        # backend-neutral distributed exact selector
│   │   ├── routing.py           # owner/demand planning and deduplication
│   │   ├── transport.py         # differentiable CP communication
│   │   ├── absorbed_mla.py      # latent sparse attention executor
│   │   └── policy.py            # pull/push/allgather fallback selection
│   └── integrations/
│       └── megatron/
│           ├── __init__.py
│           ├── contracts.py          # tagged execution request/result/capabilities
│           ├── layout_provider.py    # CPLayoutProvider protocol + Magi provider
│           ├── runtime_context.py    # microbatch-scoped CPRuntimeContext
│           ├── execution_backend.py  # AttentionExecutionBackend protocol
│           ├── expanded_mla.py       # PR #331 expanded+dense backend
│           ├── absorbed_reference.py # absorbed MLA/DSA native reference bridge
│           ├── dsa_backend.py        # absorbed+exact-top-k backend（后续实现）
│           └── spec_adapter.py       # ModuleSpec semantic prep/finalize 注入
├── exps/
│   └── megatron_attention/
│       ├── __init__.py
│       ├── config.py                 # AttentionBenchmarkCase + validation
│       ├── batch.py                  # GlobalBatch/BatchFactory/varlen oracle
│       ├── runner.py                 # 唯一 benchmark 入口
│       ├── timing.py                 # wall/CUDA event/NVTX/max-rank timing
│       ├── metrics.py                # correctness/memory/comm metrics
│       ├── provenance.py             # source/env/topology manifest
│       ├── adapters/
│       │   ├── base.py               # BenchmarkBackend protocol
│       │   ├── megatron_native.py
│       │   └── magi.py
│       ├── models/
│       │   ├── attention_stack.py
│       │   └── hybrid_proxy.py       # 3×GDN + 1×MLA unit
│       └── cases/
│           ├── smoke_mla_cp1.json
│           ├── smoke_mla_cp4.json
│           ├── smoke_packed_cp2.json
│           ├── hybrid_3to1_cp1.json
│           └── mla_dsa_cp4.json
├── tests/
│   ├── test_sparse/
│   │   ├── test_exact_topk.py
│   │   ├── test_routing.py
│   │   └── test_absorbed_mla.py
│   └── test_megatron_integration/
│       ├── test_config.py
│       ├── test_batch_factory.py
│       ├── test_layout_contract.py
│       ├── test_execution_contract.py
│       ├── test_native_adapter.py
│       ├── test_magi_adapter.py
│       ├── test_hybrid_recipe.py
│       └── test_mla_dsa_contract.py
├── scripts/
│   └── megatron_attention/
│       ├── probe_environment.sh
│       ├── run_correctness_matrix.sh
│       └── run_performance_matrix.sh
├── patches/
│   └── megatron/
│       ├── 0001-cp-runtime-context-seam.patch
│       └── 0002-mla-dsa-cp-backend-seam.patch
└── reports/
    └── megatron_attention/
        └── <run_id>/<case_name>/
            ├── result.json
            ├── rank*.log
            ├── environment.json
            └── optional profile artifacts
```

约束：benchmark 和可复用 adapter 全部落在 MagiAttention。`/Users/fankun/kernel/megatron` 只作为 pinned oracle；对 Megatron 的必要改动先保存为最小 patch series，在隔离 worktree 验证，不直接长期维护 fork。LongAttention 只允许读取并迁移通用逻辑，迁移后不双写。

## 14. 必须冻结的数据与接口契约

### 14.1 `GlobalBatch`

```python
@dataclass(frozen=True)
class GlobalBatch:
    tokens: Tensor
    labels: Tensor
    loss_mask: Tensor
    position_ids: Tensor
    valid_token_mask: Tensor
    global_token_ids: Tensor
    document_ids: Tensor
    positions_in_document: Tensor
    cu_seqlens_q: Tensor | None
    cu_seqlens_kv: Tensor | None
    cu_seqlens_q_padded: Tensor | None
    cu_seqlens_kv_padded: Tensor | None
    seed: int
    recipe_hash: str
```

不变量：同一 case 的 Native/Magi 两次 materialize 必须逐元素一致；padding token 的 `global_token_ids=-1`；任何 packed attention/indexer legality 由 `document_ids` 和 `positions_in_document` 判定，不能只根据 flat offset。

### 14.2 `LayerRecipe`

```python
@dataclass(frozen=True)
class LayerRecipe:
    pattern: tuple[str, ...] = (
        "linear_proxy",
        "linear_proxy",
        "linear_proxy",
        "mla",
    )
    repeats: int = 1
    attention_mode: Literal["dense_mla", "mla_dsa"] = "dense_mla"
    attention_representation: Literal[
        "expanded_qkv", "absorbed_latent"
    ] = "expanded_qkv"
```

约束：`attention_mode=mla_dsa` 只改变 pattern 中 `mla` slot 的选择/执行模块，并要求 `attention_representation=absorbed_latent` 才能形成 headline；不得增加第五层、不得将 LinearProxy 误替换为 DSA。

### 14.3 `AttentionBenchmarkCase`

必须包含：schema version、case name、architecture、layer recipe、`attention_mode`、`attention_representation`、model dims、DSA config、sequence/packed recipe、TP/CP/PP/DP、backend/layout policy、harness mode、dtype、seed、warmup/iterations、correctness envelope、loss policy、claim scope 和 expected capability flags。Validation 必须在启动 `torch.distributed` 前完成。

`attention_mode` 决定 selection：`dense_mla -> dense`，`mla_dsa -> exact_topk`。`attention_representation` 独立决定 `expanded_qkv | absorbed_latent`。Model dims 必须分别保存 Megatron 的 `qk_nope_head_dim` 与 `qk_pos_emb_head_dim`；execution Q/K dim 是两者之和，不能把总维度 192 直接写回 Megatron 的 `qk_head_dim` 配置。Result 必须同时保存 representation 与 selection，禁止仅写模糊的 `mla=true`。

### 14.4 `CPLayoutPlan`、`PreparedBatch` 与 `CPRuntimeContext`

Provider 必须把“全局、可序列化的物理 layout”与“当前 stage/process 的 runtime binding”分开：

```python
@dataclass(frozen=True)
class TokenSpan:
    global_start: int
    physical_length: int
    valid_length: int
    document_id: int
    position_start: int

@dataclass(frozen=True)
class RankPartition:
    spans_in_local_order: tuple[TokenSpan, ...]

@dataclass(frozen=True)
class CPLayoutPlan:
    schema_version: int
    plan_id: str
    batch_recipe_hash: str
    cp_size: int
    rank_partitions: tuple[RankPartition, ...]
    attention_pattern_ids: tuple[str, ...]
    solver_name: str
    solver_version: str
    solver_config: dict[str, object]
```

有序 `TokenSpan` 可以表示 contiguous、zigzag 和 Magi non-contiguous chunk layout；`physical_length/valid_length/document_id` 明确 padding 与 packed boundary。实现允许使用等价的 compact range encoding，但必须能确定性展开为逐 token `local_to_global`，并让不同 PP stages 对同一 plan 得到相同 hash。

```python
@dataclass
class PreparedBatch:
    local_tokens: Tensor
    local_labels: Tensor | None
    local_loss_mask: Tensor | None
    local_position_ids: Tensor
    packed_seq_params: object | None
    runtime_context: CPRuntimeContext | None
    local_to_global: Tensor

@dataclass
class CPRuntimeContext:
    backend: str
    layout_plan_id: str
    process_groups: object
    dispatch_key: object | None
    attention_keys: dict[str, object]
    local_to_global: Tensor
    global_to_owner: Tensor
    local_position_ids: Tensor
    packed_metadata: object | None
    local_attention_metadata: object
    microbatch_id: int
    pipeline_stage: int
    virtual_pipeline_stage: int | None
    loss_policy: str
```

`packed_metadata` 至少显式包含 `cu_seqlens`、`cu_seqlens_padded`、`valid_token_mask`、`document_ids` 和 `positions_in_document`。其中 physical dispatch/position 长度以 `cu_seqlens_padded` 为准，loss 与 DSA legality 以 logical metadata 和 valid mask 为准；两者不一致时禁止隐式删除 padding。

长期 facade 与 microbatch context 分离：

```python
class MagiIntegration(Protocol):
    def capabilities(self) -> object: ...
    def validate_case(self, case) -> None: ...
    def bind(
        self,
        plan: CPLayoutPlan,
        packed_metadata,
        local_attention_metadata,
        process_groups,
        microbatch_identity,
    ) -> CPRuntimeContext: ...
```

`MagiIntegration` 不保存“当前 microbatch/recent key”；runner、PP schedule 与 recompute 必须持有并显式传递返回的 context。

Provider protocol：

```python
class CPLayoutProvider(Protocol):
    def plan(self, global_batch_metadata, case) -> CPLayoutPlan: ...
    def bind(
        self,
        plan: CPLayoutPlan,
        local_attention_metadata,
        process_groups,
        microbatch_identity,
    ) -> CPRuntimeContext: ...
    def dispatch(self, tensor, context, *, pad_value) -> Tensor: ...
    def combine(self, tensor, context) -> Tensor: ...
```

Stage 0 对 tokens 执行 `dispatch`，last stage 对 labels/loss mask 执行同一 mapping；middle stages只 `bind` plan 并直接消费上游传来的 CP-local activation。Native adapter 可以不创建 Magi key，但必须填同等语义的 `local_to_global`。PP stages 只共享 `CPLayoutPlan` 或其可验证 compact encoding，不传 Python runtime key。

第一版由 first PP stage 为每个 microbatch 生成 compact plan，并沿对应 PP lane 传播 `plan + microbatch_id + plan_id`；所有 stage 在本地 `bind()` 后在 PP group 内 all-gather/比对 plan hash。若某个 stage 已从数据管线获得等价 metadata，也允许确定性重建，但 hash 不一致必须立即失败。Recompute 必须复用原 microbatch 的 plan，不能重新运行 solver 得到潜在不同 layout；plan 生成与传播开销计入 `layout/plan` headline timing。

### 14.5 `BenchmarkBackend` protocol

```python
class BenchmarkBackend(Protocol):
    def build(self, case, process_groups, shared_state) -> ModuleBundle: ...
    def prepare_batch(self, global_batch, case) -> PreparedBatch: ...
    def forward(self, modules, prepared_batch, case) -> BackendOutput: ...
    def compute_loss(self, output, prepared_batch, case) -> Tensor: ...
    def canonicalize_for_check(self, output, prepared_batch) -> Tensor: ...
    def communication_counters(self) -> dict[str, int | float | None]: ...
```

两端必须通过同一 protocol 进入 runner。Runner 禁止用 `if backend == ...` 复制两套完整训练流程；backend-specific 分支只存在于 adapter 内。

### 14.6 `AttentionExecutionBackend` typed request/result

Stable contract 使用 tagged union，不使用一个充满 optional fields 的大字典：

```python
@dataclass(frozen=True)
class ExpandedMLARequest:
    representation: Literal["expanded_qkv"]
    query: Tensor
    key: Tensor
    value: Tensor
    attention_mask: object | None
    packed_metadata: object | None
    runtime_context: CPRuntimeContext
    softmax_scale: float

@dataclass(frozen=True)
class AbsorbedMLARequest:
    representation: Literal["absorbed_latent"]
    absorbed_query: Tensor
    latent_kv: Tensor
    positional_key: Tensor | None
    up_v_weight: Tensor
    attention_mask: object | None
    packed_metadata: object | None
    runtime_context: CPRuntimeContext
    softmax_scale: float

@dataclass(frozen=True)
class ExactTopKContract:
    topk: int
    causal: bool
    document_local: bool
    scoring_semantics_id: str
    tie_break: Literal["score_then_global_token_id"]
    score_dtype: str

@dataclass(frozen=True)
class AbsorbedDSARequest:
    mla: AbsorbedMLARequest
    indexer_query: Tensor
    indexer_key: Tensor
    indexer_weights: Tensor | None
    topk_contract: ExactTopKContract
    query_global_ids: Tensor
    kv_global_ids: Tensor
    document_ids: Tensor
    valid_token_mask: Tensor
    index_share_input: object | None

AttentionExecutionRequest = (
    ExpandedMLARequest | AbsorbedMLARequest | AbsorbedDSARequest
)
```

这些 request 是一次 forward invocation 的只读 view：其中 tensor 保留 Megatron autograd graph；`up_v_weight` 等字段是 borrowed parameter tensor，不会被 Magi 注册为 `nn.Parameter`。精确 tensor shape/layout 从 pinned Megatron ModuleSpec 的 focused contract tests 生成并冻结，不能凭文档猜测后硬编码。

Result 统一为：

```python
@dataclass
class AttentionExecutionResult:
    local_output: Tensor
    output_global_ids: Tensor
    v_up_applied: bool
    topk_global_ids: Tensor | None
    topk_scores: Tensor | None
    topk_valid_counts: Tensor | None
    indexer_aux: object | None
    communication: dict[str, int | float | None]
    communication_takeover: bool
    backend_aux: object | None
```

不变量：

- `local_output` 必须按 request 的 query-owner 顺序返回；内部 Q push/KV pull layout 不得泄漏给 caller。
- Expanded backend 返回 `v_up_applied=True`；absorbed backend 可选择在 kernel 内完成 V-up，并通过该 flag 让 Megatron finalizer 只执行一次。
- DSA result 的 top-k IDs 使用 global token IDs，padding 为无效项并由 `topk_valid_counts` 区分；稳定 tie-break 由 contract 冻结。
- `indexer_aux` 只携带 Megatron finalization 所需的 differentiable statistics，不保存模型参数或跨 step runtime state。
- `communication_takeover=false` 表示 reference/fallback，不能计入 Magi 性能价值结论。

Backend protocol 与能力声明：

```python
@dataclass(frozen=True)
class AttentionExecutionCapabilities:
    representation_selection_pairs: frozenset[tuple[str, str]]
    qkv_formats: frozenset[str]
    dtypes: frozenset[str]
    supports_packed_padding: bool
    supports_backward: bool
    supports_exact_topk: bool
    supported_cp_sizes: tuple[int, ...] | Literal["any"]
    supports_tp: bool
    supports_pp_stage_local_runtime: bool

@dataclass(frozen=True)
class UnsupportedExecution:
    reason_code: str
    detail: str

class AttentionExecutionBackend(Protocol):
    def capabilities(self) -> AttentionExecutionCapabilities: ...
    def execute(
        self, request: AttentionExecutionRequest
    ) -> AttentionExecutionResult | UnsupportedExecution: ...
```

`AttentionExecutionCapabilities` 至少声明 representation/selection pairs、SBHD/THD、packed padding、backward、CP sizes、dtype、TP/PP compatibility 和是否 exact top-k。Adapter 在 distributed 初始化前校验 capability。Stable request 不包含 Megatron module、native fallback callable 或私有 forward；backend 返回 `UnsupportedExecution` 后，由 adapter 在外层执行 Megatron native fallback 并记录原因。

### 14.7 `BenchmarkResult`

JSON 顶层固定为：`schema_version`、`status`、`case`、`source`、`environment`、`execution`、`correctness`、`latency`、`throughput`、`memory`、`communication`、`artifacts`、`limitations`。`execution` 固定记录 representation、selection、backend capability snapshot、fallback reason、`v_up_applied` 与 `communication_takeover`。`status` 只能是 `pass | fail | skipped | unsupported`；任何 `skipped/unsupported` 必须带机器可读 reason。

## 15. 实施任务卡

以下任务卡是后续中等推理强度模型的唯一执行顺序。除非任务卡明确允许，不得扩大修改范围；每张卡结束都要保存产物并更新总报告。遇到停止条件时，不得通过降低正确性标准继续。

里程碑映射：S0 Scaffold=`T00–T06`；S1 Native/Magi Expanded Dense Golden=`T07–T09`；S2A Megatron Integration + Conservative Absorbed DSA=`T10–T11`；S2B Distributed DSA Value=`T12–T13`；S3 3:1 Hybrid=`T14`；S4 Internal KDA/Final Value=`T15`。每个里程碑结束必须单独出报告，不能把后续结果反向替代前一 Gate 的证据。S0 中的 3:1 build/CP1 smoke 只验证骨架可表达该 recipe，不代表 Hybrid 功能里程碑提前完成。

### T00：冻结源状态与用户改动

- **依赖**：用户确认执行。
- **允许写入**：`reports/megatron_attention/t00_source_state/`；不得修改源码。
- **步骤**：记录本地/远端 Magi 与 Megatron branch、HEAD、status、submodule SHA；对所有 dirty/untracked 用户文件保存 SHA-256；记录 host alias 和路径映射；确认无遗留 GPU/test/sync 进程。
- **执行要求**：所有远端命令必须使用 `remote-ssh-exec-ant-aistudio` skill；不得直接写 `.git`，不得 stash/reset/clean。
- **产物**：`source_state.json`、`dirty_file_hashes.json`、`remote_state.log`。
- **验收**：本地 `magic` 与远端 `magic` HEAD 一致；Megatron baseline SHA 一致；用户文件哈希表完整。
- **停止条件**：本地/远端同一文件双向变化、HEAD 分叉、未知 GPU worker 或路径不匹配。先报告，不自动覆盖。

### T01：PR #331 clean-worktree preflight

- **依赖**：T00。
- **允许写入**：临时 clean worktree、`reports/.../t01_pr331/`；确认前不得改活动 `magic`。
- **步骤**：fetch PR #331 exact head；验证 commit `99f836e...`；在 temp branch 应用；记录 changed paths、submodule change、patch-id；与 T00 dirty paths 求交集；运行 `git diff --check` 和 Python compile；审计 192/128 forward/backward 路径。
- **产物**：`pr331_files.txt`、`overlap.json`、`patch_id.txt`、`preflight.md`。
- **验收**：补丁可从 `529fb0a...` 重放；所有冲突有明确处理；用户文件 intersection 为空，或存在时有逐文件保留方案并等待人工确认。
- **停止条件**：PR head 漂移、patch 无法重放、覆盖用户修改、flash-attention submodule commit 不可取得。不得改用 #326 混补。

### T02：建立唯一远端 benchmark 环境

- **依赖**：T01 preflight 通过；不要求先将补丁合入活动分支。
- **允许写入**：项目内 `/root/work/MagiAttention/.venv`、环境探针脚本和报告；不得修改 `/opt/conda` 或已有 LongAttention 环境。该路径是用户在环境预检阶段对原计划路径的显式覆盖。
- **步骤**：从已验证 Python 3.12 基座复制隔离环境；解析 Magi/#331 与 Megatron pinned dependencies；安装最小依赖；验证同一解释器 import Torch、TE、CuTeDSL、Magi、Megatron；运行 BF16 matmul、NCCL CP2 collective、FA4 192/128 CP1 smoke。
- **目标命令形状**：`/root/work/MagiAttention/.venv/bin/python ...` 和同目录 `torchrun`；所有后续结果必须记录该 executable path。
- **产物**：`environment.json`、`pip_freeze.txt`、`gpu_topology.txt`、`import_smoke.json`。
- **验收**：Native 与 Magi 由同一 Python/Torch/CUDA/TE/NCCL stack运行；4 GPU 拓扑取证完整；BF16/NCCL/FA4 smoke 成功。
- **停止条件**：只能通过两个不同 Torch 环境运行两端、需要污染系统环境、关键 binary ABI 不一致。不得形成跨环境性能对比。

### T03：实现 schema、contracts 与静态 dry-run

- **依赖**：T00；可与 T02 的依赖下载阶段并行，但本任务不运行 GPU。
- **允许修改**：`exps/megatron_attention/config.py`、`magi_attention/integrations/megatron/contracts.py`、对应 tests/cases。
- **步骤**：实现 14 节全部 dataclass/protocol及薄 `MagiIntegration` facade；写 JSON loader、schema version、logical/physical packed metadata、`expanded_qkv/absorbed_latent × dense/exact_topk` capability validation；创建 5 个最小 cases；支持 `--dry-run --print-effective-config`。
- **本地验收命令**：`python -m exps.megatron_attention.runner --config ... --dry-run`；`pytest tests/test_megatron_integration/test_config.py`。
- **产物**：effective config JSON、schema 文档片段、unit-test log。
- **验收**：无 CUDA/Megatron/Magi import 也能解析所有 cases；非法 3:1 pattern、非法 TP×CP×PP、DSA 使用非 absorbed headline representation、QK/V dim/dtype/loss/backend capability 缺失在启动 distributed 前报清晰错误；三个 request tag 的 unit tests 能拒绝字段串用；facade 不允许保存 recent/current key。
- **停止条件**：schema 需要直接 import Megatron/Magi 类型、runner 内出现 backend-specific tensor 逻辑。

### T04：实现 `BatchFactory` 与 packed-varlen oracle

- **依赖**：T03。
- **允许修改**：`batch.py`、`test_batch_factory.py`、case fixtures。
- **步骤**：实现 deterministic SBHD/THD global batch；同时生成 logical `cu_seqlens` 与 physical `cu_seqlens_padded`；支持 logical/padded lengths 不同；生成 document/global IDs、valid mask、position、local labels/loss mask；实现 native/Magi materialize equality check；实现 causal/document legality oracle、local-loss token-count oracle 和 round-trip permutation oracle。
- **必要 fixtures**：dense single sequence；`[1024,512,256,128,64,64]`；非整除且带 padding 的小 fixture，例如 logical `[13,9,20,14]`、padded `[16,12,20,16]`；boundary-heavy 大量短文档。
- **产物**：fixture JSON、batch hashes、oracle report。
- **验收**：同 seed 输出逐元素一致；padding IDs 为 -1；physical position/dispatch 长度与 padded flat tensor 一致；loss numerator/count 只覆盖 logical valid tokens；round-trip 恢复原顺序；任何跨文档 KV selection 被测试捕获。
- **停止条件**：要求每个文档长度整除 `2×CP`、通过删除短文档规避不均匀 layout、仅比较 tensor shape 不比较 token identity。

### T05：实现唯一 runner、timing 与 provenance 骨架

- **依赖**：T03、T04。
- **允许修改**：`runner.py`、`timing.py`、`metrics.py`、`provenance.py`、base adapter。
- **步骤**：实现状态机 `validate -> init groups -> materialize -> build -> correctness -> warmup -> measure -> report`；加入两个 harness；headline timeline 从 batch generation 开始，独立记录 `batch/materialize`、`THD pack`、`layout/plan`、`dispatch`、`forward`、`local loss`、`backward`；CPU wall clock、CUDA events、NVTX ranges、rank-local samples 和 max-rank aggregation；实现 atomic JSON write 与失败结果落盘。
- **计时规则**：barrier 只用于样本起点对齐；每 rank 本地计时，样本结束后再 all-reduce MAX；compile/JIT 单列；失败时保留 traceback 和已完成阶段。
- **产物**：dry-run result、mock-adapter result、JSON schema example。
- **验收**：mock native/Magi adapter 使用同一 runner；两个 harness 复用同一 batch/schema/timeline，one-layer-training E2E 只额外包含 loss/backward/optimizer-zero 边界；`calc_attn` breakdown 不得冒充 headline；未实现字段为 `null + reason`；中途异常仍产生 `status=fail` JSON。
- **停止条件**：两端各有一份 runner、只打印日志不保存结构化结果、性能 sample 内混入结果文件 IO。

### T06：实现真实 `MegatronNativeAdapter` 和 Scaffold Gate

- **依赖**：T02、T05。
- **允许修改**：native adapter、attention/hybrid model builder、focused tests；不得改 Megatron baseline checkout。
- **步骤**：用 pinned Megatron public `ModuleSpec` 构造 dense MLA 与 native MLA+DSA；从 `GlobalBatch` 走 native CP preprocessing、Embedding、真实 attention、固定 gradient或 local LM-head loss；为 MLA builder 暴露可测试的 semantic preparation/execution/finalization 边界而不复制 forward；构造 `GDN,GDN,GDN,MLA` 单元；实现 shared state-dict clone；补充 rearranged Q/K/V stride/contiguous/dtype assert 与 non-contiguous backward/NaN regression。
- **远端 smoke**：CP1 dense MLA 两个 harness；CP1 3:1 Hybrid forward/backward；CP1 MLA+DSA build/fallback smoke；CP2 small packed native smoke。
- **产物**：每个 case 的 `result.json`、state-dict key manifest、rank logs。
- **Scaffold Gate 验收**：12 节的 8 条全部满足；runner 能切换 architecture/attention mode/harness；真实 Native path 通过；Magi adapter 可以是明确标注的 reference/fallback；尚无性能结论。
- **停止条件**：复制 Megatron attention forward、伪造 QKV 代替 global batch、Hybrid recipe 不是 3:1、MLA+DSA 构造改变 MLA checkpoint keys。

### T07：冻结 Native 黄金矩阵

- **依赖**：S0/T06。
- **允许修改**：cases、matrix script、报告；只允许修复 benchmark 基础设施 bug，不做 Magi 优化。
- **步骤**：运行与 PR #331 对齐的 expanded dense MLA CP1/2/4 SBHD/THD；逐 case 枚举 native `p2p/all_gather/a2a/hierarchical` 的实际可用子集；另行运行 absorbed MLA+DSA native AllGather oracle；保存 representation-tagged output/grad/loss golden 和 nsys baseline。额外运行 `C0/C1` standard-attention variable-length control，验证 runner 能公平计入 dynamic CP scheduler/group lifecycle；`M2/D3` 在当前版本记录为明确 unsupported，不通过换模型绕过。
- **产物**：`native_matrix.json`、最快 baseline selection table、golden tensors/hashes、nsys reports。
- **验收**：每个 headline case 有最快正确 native mode；unsupported 有原因；重复两轮 CV 与 correctness 合格；Magi package 不参与 native timing path。
- **停止条件**：选择较慢 baseline、不同 backend 使用不同 batch/environment、correctness 未过先计时。

### T08：实现 Magi expanded dense MLA adapter

- **依赖**：T01 补丁获准合入、T02、T07。
- **允许修改**：PR #331 provenance commit、Magi adapter、layout provider/runtime context、focused tests。
- **步骤**：通过 `CPLayoutProvider` 从同一 global batch 构造 Magi key；运行真实 Megatron MLA projection/RoPE，创建 `ExpandedMLARequest`；使用 TP-local head metadata、`head_dim=192/head_dim_v=128` 和 packed metadata 调用真实 `dispatch -> calc_attn`；将 `AttentionExecutionResult` 交回 Megatron output projection；根据 case policy 保持 local layout 或 combine；接入 backward。
- **首轮范围**：full causal、BF16、even shard，随后 packed varlen；不实现 arbitrary mask、PP 或 persistent Hybrid layout。
- **产物**：CP1/2/4 dense MLA parity JSON、dispatch round-trip report、communication counters。
- **验收**：从 tokens/hidden states 开始的 output/input-grad/全部 MLA param-grad 达到冻结 envelope；state-dict/parameter identity 不变；Magi backend 自身无参数；`representation=expanded_qkv`、`communication_takeover=true`；结果可追溯到 #331 patch/submodule SHA。
- **停止条件**：劫持 TE 全局 backend、裸 Magi key散落到所有模型 API、通过 undispatch vocab logits 掩盖 local-loss 问题。

### T09：运行 dense MLA 黄金 E2E 对照

- **依赖**：T08 correctness。
- **允许修改**：性能 cases、profile scripts、报告；一次只允许一个 profile-driven 变量变化。
- **步骤**：运行 16K/32K/64K/128K、CP1/2/4 可行组合；两个 harness 全部测；A/B 顺序交替；保存 max-rank p50/p10/p90、memory、payload；先 nsys，只有热点 kernel 再 ncu。headline 先比较 `M0/M1` 固定 CP；另行报告 `C0/C1` 的 dynamic CP scheduler 收益，禁止交叉归因。只有固定 CP Magi 与 dynamic CP scheduler 各自通过后，才允许追加 `M3` 组合实验。
- **产物**：`dense_mla_value_report.md/json`、profiles、break-even table。
- **验收**：与 T07 每 case 最快 native baseline 比较；compile time 与 steady state 分离；报告 planning/dispatch/comm/kernel/backward 分解。
- **停止条件**：只报告 kernel 时间、忽略 combine、使用不同 batch 或只挑有利 CP mode。Dense 未加速可以如实结束本卡，不阻塞 MLA+DSA 研究。

### T10：验证 Megatron 优雅集成与 PP runtime lifecycle

- **依赖**：T09。T09 未达到性能目标不阻塞本卡，只要 dense golden、correctness 和性能分解已经完整落盘。
- **允许修改**：Magi integration package、隔离 Megatron worktree、`0001` patch、integration tests。
- **步骤**：先在 expanded dense MLA 上分别实现 backend-neutral `CPLayoutProvider/CPRuntimeContext` 与 `AttentionExecutionBackend` seam；ModuleSpec 保留 Megatron semantic preparation/finalization，只替换 execution；first PP stage dispatch tokens；所有 stages 从同一 serializable layout plan 绑定本地 CP group/key；last stage dispatch labels/loss mask 并做 local loss。两条 seam 随后组合承载 MLA+DSA，不依赖 Hybrid stack。
- **覆盖**：TP-local Q/KV heads；PP2×CP2；多 microbatch；recompute；packed metadata；distributed checkpoint；provider absent/native fallback。
- **产物**：最小 patch、patch-replay report、PP/TP parity results、state-dict diff。
- **验收**：Megatron 不 import Magi；无 global registry/monkeypatch；checkpoint keys/optimizer parameter identity 不变；patch 可从干净 pinned SHA 重建。
- **停止条件**：Python key 跨 PP process 传输、middle stage 无 context、修改大量 GPT forward 实现、长期 fork 才能运行。

### T11：建立 native MLA+DSA oracle 与 pre-AllGather seam

- **依赖**：T07、T10。
- **允许修改**：typed DSA contract、`0002` patch、`MagiDSAReferenceBackend`、native/reference provider、tests。
- **步骤**：提取 Megatron local absorbed-MLA 与 indexer semantic preparation，构造 `AbsorbedDSARequest`；在任何 dense AllGather 前提供可选 execution seam；backend 返回 `UnsupportedExecution` 时由 adapter 原样走 native path。第一实现由 adapter 选择 Megatron AllGather reference，完整记录 top-k、indexer loss/index-share、packed ownership 和通信统计；request 内不放 fallback callable。只有 Magi 自己执行同等 AllGather 后才能标记 `communication_takeover=true`。
- **表示约束**：PR #331 的 dense MLA `Q/K=192,V=128` 支持不能直接等同于 absorbed MLA+DSA。若 Magi IndexAttn 尚不支持 absorbed query/key、implicit latent value 与 `up_v_weight`，本卡继续使用 Megatron sparse kernel/reference path，不展开 dense per-head KV 来绕过限制。
- **产物**：tagged request/result contract、parameter-ownership/state-dict manifest、fallback/reference tests、CP1/2/4 oracle JSON、patch series、`dsa_scaffold_report.md`。
- **验收**：provider on/off 的 output、top-k IDs、loss、input/parameter grads、state dict、optimizer identity 对齐；`up_v_weight` 只注册一次；Magi backend parameter count 为 0；native fallback 的 communication 与原路径一致；不要求 speedup。
- **停止条件**：复制 `DSAttention.forward`、把 indexer 参数移入 Magi、把只位于 AllGather 后的 kernel hook 当成最终 CP execution seam、改变 MLA checkpoint 参数化。

### T12：实现 exact distributed indexer baseline

- **依赖**：T11。
- **允许修改**：`magi_attention/sparse/exact_topk.py`、最小 transport primitives、DSA adapter、tests/bench cases；core sparse 模块不得 import Megatron。
- **步骤**：保持 local indexer Q；按 CP owner stream/ring indexer K；tile score；维护 running TopK；用 `(score, global_token_id)` 稳定排序；携带 document/position legality；返回 fixed-size IDs + valid count；实现 backward/indexer-loss 所需数据流。
- **第一版不优化**：允许朴素 PyTorch collective 和 reference kernels，只要求 exact、bounded memory、接口正确。
- **产物**：top-k parity matrix、peak score-buffer memory、indexer communication report。
- **验收**：CP1/2/4、SBHD/THD global IDs 与 native oracle exact；无跨文档 token；不物化完整 `[global_q, global_k]` score matrix。
- **停止条件**：使用近似 top-k、改变 scoring 算法、丢失 stable tie-break、通过 AllGather 完整 score tensor 伪装成 streaming。

### T13：实现并选择 sparse latent-MLA executor

- **依赖**：T12。
- **允许修改**：`magi_attention/sparse/{routing,transport,absorbed_mla,policy}.py`、IndexAttn/kernel adapter、profiling/tests；Megatron-specific tensor 翻译继续只在 integration adapter。
- **步骤 A**：compressed latent KV AllGather + local IndexAttn，建立 distributed sparse-compute baseline。
- **步骤 B**：selected latent-KV pull，按 owner 去重请求、AllToAll/P2P fetch、owner-return gradients。
- **步骤 C**：query push/remote partial MLA，在 KV owner 计算局部 logits/softmax statistics/output，再精确合并并返回 query owner。
- **决策数据**：unique selected coverage、owner fanout、metadata bytes、forward/backward payload、max-rank imbalance、E2E latency、memory。
- **fallback**：高 coverage/fanout 自动回退 compressed AllGather/native，fallback 选择计入 E2E。
- **产物**：三策略对照、break-even model、最终 policy、MLA+DSA value report。
- **验收**：exact top-k 不变；output/grad/loss 合格；至少两个 ultra-long case 相对 native MLA+DSA E2E +10%；payload 与峰值显存下降。
- **停止条件**：只减少计算不减少通信却宣称 DSA 完成、通信 expanded K/V、只跑 forward、只用单 GPU IndexAttn 数字。

### T14：实现 3:1 Hybrid 完整训练与 layout 验证

- **依赖**：T06、T10、T13。T06 只提供最小 build smoke；本卡才是 Hybrid 正式里程碑。
- **允许修改**：`hybrid_proxy.py`、recipe cases、training verifier/tests。
- **步骤**：构造 1 个和 2 个重复 unit：每个 unit 严格 `GDN,GDN,GDN,MLA`；pretraining mode 使用已经验证的 dense MLA；posttraining mode 将每个 unit 的同一个 MLA slot 切成已经验证的 MLA+DSA executor；LinearProxy 只接通 norm/residual/forward/backward，不优化 kernel。
- **layout v1**：GDN 保持 canonical layout；进入每个 MLA/MLA+DSA slot 前 Magi dispatch，返回后 combine；所有转换纳入 timing。
- **训练验证**：50 steps；step 25 model/optimizer checkpoint resume；activation recompute；CP2；TP2×CP2；PP2×CP2 分开运行。
- **persistent layout 研究**：只有 v1 全部通过后，才按 GDN 有序 recurrent-state contract 评估 order-preserving layout、state handoff 和一般 non-contiguous layout；不把它设为 Gate 3A/3B 的前置条件。
- **产物**：recipe manifest、training parity JSON、checkpoint manifest、layout transition counters、可选 v1/v2 对照。
- **验收**：层序、参数和调用计数证明 3:1；Native/Magi 参数初值与 optimizer state 独立但相同；pretraining/posttraining 的 loss/grad/update 轨迹合格；v2 仅在正确且 E2E +5% 时保留。
- **停止条件**：为了避免转换成本改成 1:1、把 GDN 性能当内部 KDA、未证明 state 顺序就启用 non-contiguous persistent layout、重新实现另一套 MLA+DSA。

### T15：内部 KDA 与真实模型替换

- **依赖**：T13、T14；需要用户提供内部 ModuleSpec/config。
- **允许修改**：新增 internal adapter/config（不得把内部代码写入公开 case）、最终 reports。
- **步骤**：用真实 KDA 替换 LinearProxy；保持 3:1 recipe；导入真实 dims、DSA top-k/frequency/index-share、packed distribution、TP/CP/PP；重新决定 attention-local 与 persistent layout；重跑 correctness、50-step、checkpoint 和 E2E。
- **Linear 优化边界**：只有真实 KDA 接通且整体瓶颈报告显示 Linear 占主导后，才单独建立 Linear optimization workstream；不得提前拖慢主骨架。
- **产物**：公开/内部双矩阵、最终集成 patch、value report、已知限制。
- **验收**：内部主 case 几何平均 +10%，无主 case 回退 >5%，参数/checkpoint/训练生命周期正确，Megatron patch 保持小且 backend-neutral。
- **停止条件**：用 GDN 数字代替内部 KDA 结论、内部配置未记录 provenance、为单一内部模型破坏通用 contract。

## 16. 中等推理强度模型执行规则

1. 一次只执行一张任务卡；卡内验收完成前不得开始下一卡。
2. 开始前读取本文件、AGENTS.md、目标仓库状态和该卡依赖产物；不得依赖聊天记忆代替文件证据。
3. 远程同步/执行只能使用 `remote-ssh-exec-ant-aistudio` skill；先 doctor/preview，再同步；出现 `contents changed on both sides` 立即停止。
4. 所有 GPU、pytest、benchmark、profile 验证在 `aistudio-61480188-ssctl` 远端完成；本地只做静态/dry-run。
5. 修改文件必须限制在任务卡允许范围；发现需要扩大接口时先更新设计报告并等待确认。
6. correctness 先于 performance；任何 performance JSON 必须引用已通过的 correctness artifact。
7. 不使用 `SIGKILL`；远端任务先 SIGTERM/正常退出；wrapper 提前返回时先查现有 PID/result，不重复启动。
8. 每次 commit 前核对 T00 用户文件哈希；只 stage 当前任务文件；commit/push 后远端必须对齐同 branch/HEAD。
9. 每张卡结束输出：修改文件、命令、status code、artifact 路径、通过/失败项、下一卡是否解锁。
10. `unsupported/skipped` 不是 pass；缺依赖、缺硬件或上游限制必须保留明确证据。

## 17. 最终需求—证据审计表

| 原始诉求 | 必须存在的权威证据 |
| --- | --- |
| 在 MagiAttention 上迭代 | 实现、tests、runner 均位于 `MagiAttention/magic`；LongAttention 不再双写 |
| Benchmark 从 batch 生成开始 | `batch_to_backward_wall_ms`、NVTX timeline 和 runner 状态机证明起点在 `BatchSource.next()` |
| 两种端到端 benchmark | `attention_stack` 与 `attention_to_loss` 同 case 的独立 JSON |
| 抽出真实 Megatron attention | Native adapter 的 ModuleSpec/source SHA、state-dict manifest、无复制 forward 审计 |
| PR #331 不冒充完整 MLA | E2E timeline 证明 Megatron projection/RoPE/output projection 包围 `ExpandedMLARequest`；case/result 明确记录 `expanded_qkv+dense` |
| 公平比较 Magi/Megatron | 同 batch hash、same state dict、same env、最快 native selection table |
| 3:1 Hybrid 架构 | recipe manifest、layer call counters、`GDN×3 + MLA×1` forward/backward/50-step artifacts |
| 后训练加入 DSA 且保留 MLA | `AbsorbedDSARequest`、MLA/MLA+DSA state-dict关系、exact top-k/loss/output parity |
| 参数归属不被破坏 | Native/Magi state-dict keys、parameter identity、Magi backend zero-parameter manifest、`up_v_weight` 单一注册证据 |
| Ultra-long 与 varlen | 128K/256K cases、1M feasibility、logical/padded THD document-boundary oracle |
| 优雅集成 Megatron | backend-neutral context、ModuleSpec、可重放最小 patch、TP/PP/recompute/checkpoint tests |
| 优化确实有效 | 相对每 case 最快 native 的 E2E/memory/communication value report，不使用 microkernel 替代 |
| 保护现有工作 | T00/T15 前后 dirty file hashes、branch/HEAD/sync evidence |

只有表中每一行都有直接证据，才允许把整个目标标记为完成。

## 18. 冻结的 runner CLI 与配置示例

### 18.1 唯一入口

```text
python -m exps.megatron_attention.runner \
  --config <case.json> \
  --backend <native|magi|both> \
  --harness <attention_stack|attention_to_loss> \
  --phase <dry_run|correctness|performance> \
  --output-dir <run_dir>
```

约束：`backend=both` 必须在同一 torchrun job 中顺序构建两端、从同一 immutable `GlobalBatch` 和深拷贝 state dict 开始；不得启动两个环境独立运行后再拼结果。`performance` 自动查找并引用同 case/backend/harness 的 correctness artifact，不存在则拒绝启动。

### 18.2 3:1 Hybrid case 示例

```json
{
  "schema_version": 1,
  "case_name": "hybrid_3to1_dense_mla_cp4_16k",
  "architecture": "hybrid_linear",
  "claim_scope": "headline",
  "layer_recipe": {
    "pattern": ["linear_proxy", "linear_proxy", "linear_proxy", "mla"],
    "repeats": 1,
    "attention_mode": "dense_mla",
    "attention_representation": "expanded_qkv"
  },
  "linear_backend": "megatron_gdn",
  "model": {
    "hidden_size": 7168,
    "num_attention_heads": 128,
    "num_query_groups": 128,
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "qk_head_dim": 128,
    "qk_pos_emb_head_dim": 64,
    "v_head_dim": 128,
    "dsa_indexer_n_heads": 64,
    "dsa_indexer_head_dim": 128,
    "dsa_indexer_topk": 2048
  },
  "sequence": {
    "format": "sbhd",
    "global_tokens": 16384,
    "micro_batch_size": 1,
    "logical_lengths": [16384],
    "padded_lengths": [16384]
  },
  "parallel": {
    "tensor": 1,
    "context": 4,
    "pipeline": 1,
    "data": 1
  },
  "layout": {
    "native": "megatron_zigzag",
    "magi": "attention_local"
  },
  "runtime": {
    "dtype": "bf16",
    "seed": 1234,
    "native_cp_modes": ["p2p", "all_gather", "a2a"],
    "magi_kernel": "ffa_fa4",
    "loss_policy": "local",
    "warmup": 5,
    "iterations": 20
  },
  "correctness": {
    "relative_l2_max": 0.005,
    "cosine_min": 0.999,
    "require_exact_topk": false
  }
}
```

切换到后训练只允许改变：

```json
{
  "layer_recipe": {
    "attention_mode": "mla_dsa",
    "attention_representation": "absorbed_latent"
  },
  "correctness": {"require_exact_topk": true}
}
```

实际 loader 应做深度 merge；不得因此丢失 3:1 pattern 或 model fields。

### 18.3 标准命令形状

本地只做静态验证：

```bash
python -m exps.megatron_attention.runner \
  --config exps/megatron_attention/cases/hybrid_3to1_cp1.json \
  --backend both --harness attention_stack --phase dry_run \
  --output-dir /tmp/magi_megatron_dry_run

python -m compileall magi_attention/integrations/megatron exps/megatron_attention
pytest -q tests/test_megatron_integration/test_config.py \
  tests/test_megatron_integration/test_batch_factory.py
```

远端 correctness（命令由 remote skill 包装执行）：

```bash
RUN_ID=YYYYMMDD_HHMMSS
/root/work/envs/magi-megatron-bench/bin/torchrun \
  --standalone --nproc-per-node=4 \
  -m exps.megatron_attention.runner \
  --config exps/megatron_attention/cases/smoke_mla_cp4.json \
  --backend both --harness attention_stack --phase correctness \
  --output-dir reports/megatron_attention/${RUN_ID}/smoke_mla_cp4
```

远端 performance：

```bash
RUN_ID=YYYYMMDD_HHMMSS
/root/work/envs/magi-megatron-bench/bin/torchrun \
  --standalone --nproc-per-node=4 \
  -m exps.megatron_attention.runner \
  --config exps/megatron_attention/cases/hybrid_3to1_cp4_16k.json \
  --backend both --harness attention_to_loss --phase performance \
  --output-dir reports/megatron_attention/${RUN_ID}/hybrid_3to1_cp4_16k
```

Matrix scripts只负责枚举已冻结 cases，并逐 case 调用同一入口；不得在 shell 中实现另一套 timing/correctness 逻辑。

### 18.4 Profiling 规则

- Runner 为 `batch/materialize`、`layout/plan`、`dispatch`、`embedding`、`projection`、`comm`、`kernel`、`loss`、`backward`、`combine` 建立固定 NVTX ranges。
- Nsys 仅在 correctness 通过后运行，使用 CUDA profiler range 捕获 steady-state iteration；保存 `.nsys-rep` 与 stats CSV。
- NCU 只针对 nsys 确认的单个热点 kernel；一次只改一个 kernel/config 变量，并保存基线/实验两份 `.ncu-rep`。
- NCCL payload 以 adapter transport 的逻辑 bytes/call counters 为主，并用 nsys NCCL trace 交叉验证；不得用理论公式替代实际调用计数。
