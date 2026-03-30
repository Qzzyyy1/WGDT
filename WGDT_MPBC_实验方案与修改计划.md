# WGDT_MPBC 实验方案与修改计划

## 1. 文档目的

本文档用于统一后续研究与开发方向，避免继续在 `UOT_OSDA` 现有复杂分支上进行低收益的反复调参。

本文档主要回答 5 个问题：

1. 当前方案为什么不再继续沿原路线修补。
2. 新方案准备借鉴哪些论文思想，以及借鉴点是什么。
3. 新方案主要解决哪些核心难点。
4. 模型结构、损失函数、训练流程准备如何设计。
5. 代码层面准备修改哪些文件，按什么顺序推进。

---

## 2. 当前阶段结论

### 2.1 已验证结论

结合最近多轮实验，当前可以较为明确地得到以下结论：

- `UOT_OSDA` 这条主线在加入 `radius / barycenter / classwise radius / eval EMA / dual boundary` 等机制后，虽然局部稳定性有所改善，但 **HOS 上限始终上不去**。
- `source_oa` 在适配阶段长期保持较高，说明问题 **不是源域灾难性遗忘**。
- 当前性能瓶颈更像是：
  - 决策几何不统一；
  - 单原型难以覆盖类内多模态；
  - unknown 边界并不是通过强判别信号学出来的；
  - 多分支同时参与 unknown 判定，导致优化方向不一致。

### 2.2 为什么不再继续主修 `UOT_OSDA`

当前 `UOT_OSDA` 中同时存在以下多个判别因素：

- `UOT`
- `prototype`
- `anchor`
- `tuplet`
- `radius`
- `EMA`
- 若继续打开，还包括 `barycenter`

这些分支并未围绕 **同一个开集判别空间** 设计，导致模型容易出现：

- source 很稳，但 target HOS 天花板低；
- 某些 known 类极好，某些 known 类极差；
- 有时能冲高，但后期会回落；
- 有时看似稳定，但上限明显不如 WGDT。

因此，当前推荐策略不是继续在 `UOT_OSDA` 上叠加小补丁，而是：

> 回到 `WGDT` 的核心思想，用更少但更强的结构替换现有“多分支打架”的设计。

---

## 3. 新方案名称与总目标

## 3.1 方案名称

建议新方案命名为：

`WGDT_MPBC`

其中：

- `MP` = `Multi-Prototype`
- `BC` = `Boundary Contrastive`

中文可写为：

- **基于多原型与边界对比校准的 WGDT 开放集域适应模型**

### 3.2 总目标

在保留 `WGDT` 核心优点的基础上，重点解决以下两个痛点：

1. **known 类内部结构建模太粗糙**
   - 目标：提升低准确率 known 类表现。

2. **unknown 边界学习过弱**
   - 目标：提高 HOS 上限，并减轻后期性能回落。

---

## 4. 借鉴论文与借鉴点

下面列出新方案准备借鉴的主要论文思想。这里的“借鉴”不是照抄结构，而是提取其中对当前任务最有价值的设计原则。

### 4.1 WGDT（当前最强本地 baseline）

**借鉴点：**

- `anchor / CAC` 提供清晰的类几何结构。
- `gamma` 天然适合作为 open-set score。
- 训练期间一直保留 source 监督，有利于稳定优化。

**保留内容：**

- `feature_encoder`
- `classifier`
- `anchor / CAC`
- source 持续监督训练

---

### 4.2 Class Anchor Clustering, WACV 2021

- 标题：`Class Anchor Clustering: A Loss for Distance-Based Open Set Recognition`
- 链接：<https://openaccess.thecvf.com/content/WACV2021/html/Miller_Class_Anchor_Clustering_A_Loss_for_Distance-Based_Open_Set_Recognition_WACV_2021_paper.html>

**借鉴点：**

- 已知类应在 logit / anchor 空间形成紧致、可分的类簇。
- 开集识别的基础不是“后处理阈值”，而是训练期就塑造好类几何。

**对本方案的启发：**

- `anchor/CAC` 仍然应是主干，而不是辅助模块。
- 类别语义应优先由 anchor 空间承担。

---

### 4.3 Dual-Branch Hyperspectral Open-Set Classification with Reconstruction–Prototype Fusion, Remote Sensing 2025

- 链接：<https://www.mdpi.com/2072-4292/17/22/3722>

**借鉴点：**

- 高光谱开放集分类中，原型分支对 known / unknown 分离有价值。
- 但辅助分支更适合做结构增强与校准，而不适合作为最终裁决的唯一来源。

**对本方案的启发：**

- prototype 分支要保留，但不再让其独立支配最终判决。
- prototype 应与 anchor 主干形成互补，而不是形成第二套竞争判决机制。

---

### 4.4 Open Set Domain Adaptation via Known Joint Distribution Matching and Unknown Classification Risk Reformulation, TNNLS 2026

- 链接：<https://pubmed.ncbi.nlm.nih.gov/41533626/>

**借鉴点：**

- OSDA 的核心不是盲目对齐，而是同时处理 known 对齐与 unknown 风险。
- unknown 分类风险应被显式建模，而不是当成阈值后处理问题。

**对本方案的启发：**

- target 端必须显式学习 known-like 与 unknown-like 的边界。
- 这支持我们使用 `boundary contrastive / ranking` 而不是标量半径回归。

---

### 4.5 2024–2025 多原型 / 原型结构类工作（方向性借鉴）

当前近两年多篇工作都在反复强调：

- 单原型不足以建模复杂类别；
- 多原型更适合处理类内多模态；
- prototype consistency 对开放集和跨域泛化都有帮助。

**对本方案的启发：**

- 已知类不应继续只用单 prototype 表征。
- 应该引入 **每类多个子原型** 的设计。

---

## 5. 新方案主要解决的难点

### 5.1 难点一：个别 known 类准确率极低

**现象：**

- 当前实验中部分 known 类准确率长期远低于其他类。

**判断原因：**

- 单原型难以覆盖类内多模态分布；
- 某些类内部样本可能天然分散或包含多个子簇；
- 当用单中心逼近时，这些类更容易被误判为 unknown 或被吸向其他类。

**对应策略：**

- 为每个 known 类维护 `K=2` 或 `K=3` 个子原型；
- 用“最近子原型”而不是“类均值原型”描述类内结构。

---

### 5.2 难点二：unknown 边界学习太慢、太弱、上限低

**现象：**

- 学习型 radius 很难稳定又高效地学到强边界；
- 很多实验要么慢涨，要么波动，要么最终 HOS 上限明显偏低。

**判断原因：**

- 标量边界回归不是强判别学习；
- unknown 边界没有通过 known-like / unknown-like 对比直接被拉开。

**对应策略：**

- 直接在 target 上构造：
  - `pseudo-known`
  - `boundary-uncertain`
  - `pseudo-unknown`
- 用排序/对比损失直接学习边界。

---

### 5.3 难点三：多分支打架，优化不统一

**现象：**

- source 指标好看，但 target HOS 始终不够理想；
- 同时存在多个 unknown 判决来源时，优化目标不一致。

**对应策略：**

- 统一判决逻辑：
  - `anchor / CAC` 负责“是谁”
  - `multi-prototype consistency + gamma` 负责“是不是 unknown”
- 取消多余主判决分支。

---

## 6. WGDT_MPBC 总体结构设计

## 6.1 主干保留

以 `WGDT.py` 为核心骨架，保留：

- `feature_encoder`
- `classifier`
- `anchor`
- `CAC loss`

原因：

- 这是目前最稳定、最可信、最具解释性的几何主干。

---

### 6.2 新增模块一：Multi-Prototype Memory

对每个 known 类维护多个子原型：

- `K = 2` 作为第一版默认配置；
- 后续再做 `K=1/2/3` 消融。

**输入：**

- source 特征；
- 高置信 target-known 特征（可选，小步更新）。

**作用：**

- 建模类内多模态结构；
- 缓解已知类中“弱类塌陷”问题。

---

### 6.3 新增模块二：Boundary Contrastive Calibration

不再继续使用“learnable scalar radius regression”作为主边界学习方式。

改为基于 target 样本构造边界对比：

- `confident-known`：
  - anchor 置信高；
  - `gamma` 小；
  - prototype 一致性高。

- `pseudo-unknown`：
  - `gamma` 大；
  - prototype 一致性差；
  - 或者类间 margin 极小。

- `boundary-uncertain`：
  - 介于两者之间，用作边界过渡带。

目标：

- 让 `open_score(known)` 显著小于 `open_score(unknown)`；
- 通过 ranking / contrastive 的方式直接拉开边界。

---

### 6.4 最终 open-set score 设计

建议最终 unknown score 由两部分组成：

- `gamma_score`：来自 WGDT 的 anchor 分支；
- `proto_consistency_score`：来自多原型一致性。

建议形式：

`open_score = gamma_score + lambda * proto_consistency_score`

其中 `proto_consistency_score` 可以由以下项构成：

- 到预测类最近子原型的距离；
- 与次近异类子原型的 margin；
- 或者二者融合。

**设计原则：**

- `anchor` 负责类别语义；
- `prototype` 负责类内结构；
- `open_score` 负责 unknown 判定；
- 三者统一在一个决策逻辑里。

---

## 7. 建议去掉或降级的模块

第一版 `WGDT_MPBC-v1` 建议尽量纯化，不再保留以下复杂分支作为主路径：

- `barycenter`
- `learnable radius regression`
- `classwise radius` 系列实验逻辑
- `DANN` 主对抗分支
- `UOT` 作为最终 unknown 判决分支

### 7.1 关于 UOT 的处理建议

第一版建议：

- **直接不用 UOT 主损失。**

原因：

- 目前多轮实验已经说明，UOT 并未成功形成更强的统一决策几何；
- 继续保留它只会提高实验不确定性，削弱主线判断。

后续如果 `WGDT_MPBC-v1` 有效，再考虑把 `UOT` 作为弱辅助先验重新接回。

---

## 8. 损失函数设计

总损失建议为：

`L = L_src_cls + alpha * L_src_anchor + beta * L_src_tuplet + gamma * L_proto_src + delta * L_proto_tgt + eta * L_boundary + mu * L_consistency`

### 8.1 源域损失

- `L_src_cls`
  - 普通已知类分类损失。

- `L_src_anchor`
  - anchor/CAC 的 anchor 部分。

- `L_src_tuplet`
  - anchor/CAC 的 tuplet 部分。

- `L_proto_src`
  - source 样本与本类最近子原型对齐。

### 8.2 目标域损失

- `L_proto_tgt`
  - 仅对高置信 pseudo-known target 做子原型对齐。

- `L_boundary`
  - 核心创新项；
  - 对 known-like 与 unknown-like 样本做边界排序/对比约束。

- `L_consistency`
  - 约束 anchor 预测与 prototype 预测的一致性；
  - 避免两套判别空间彼此冲突。

---

## 9. 训练流程设计

### 9.1 阶段一：Source 预训练

目标：

- 训练稳定 backbone；
- 建立 anchor 几何；
- 初始化多原型结构。

损失：

- `L_src_cls`
- `L_src_anchor`
- `L_src_tuplet`
- `L_proto_src`

### 9.2 阶段二：Source + Target 联合适配

#### Source 端

- 持续保留 source 监督；
- 维持类几何与 source 判别能力；
- 防止 known 类被 target 干扰拉坏。

#### Target 端

- 不直接上伪标签 CE；
- 只对高置信样本做原型对齐；
- 通过 `boundary contrastive` 学 unknown 边界。

---

## 10. 版本推进计划

### 10.1 V1：极简重构版（优先实现）

目标：先验证“结构统一”是否优于当前复杂版。

**保留：**

- WGDT 主干
- 多原型 memory
- source / target prototype loss
- boundary contrastive loss
- consistency loss

**不加入：**

- UOT
- barycenter
- DANN
- learnable radius
- eval EMA 评估平滑

### 10.2 V2：轻量 target 结构增强

前提：V1 有明显收益。

可能追加：

- target 高置信样本队列
- prototype EMA refinement
- 极弱的 target structure regularization

原则：

- 不破坏 V1 的统一判决逻辑；
- 不再引入新的主裁决分支。

---

## 11. 实验方案

## 11.1 Baseline 组

用于对照：

- `WGDT baseline`
- 当前最佳旧版 `UOT_OSDA` 结果

### 11.2 新方案消融组

建议至少做以下 5 组：

1. `WGDT + Multi-Prototype`
2. `WGDT + Boundary Contrastive`
3. `WGDT + Multi-Prototype + Boundary Contrastive`
4. `WGDT + Multi-Prototype + Boundary Contrastive + Consistency`
5. `WGDT + Multi-Prototype + Boundary Contrastive + Consistency + target prototype update`

### 11.3 关键超参消融

- 子原型个数：`K = 1 / 2 / 3`
- boundary margin：`m = 0.05 / 0.1 / 0.2`
- target 高置信阈值：若需要，可做 `0.7 / 0.8 / 0.9`
- consistency 权重：小范围消融

### 11.4 主要评估指标

- `HOS`
- `unknown`
- `oa_known`
- `aa_known`
- `classes_acc`

重点观察：

- 之前掉得最惨的 known 类是否明显回升；
- 是否还能保持 unknown 检测能力；
- HOS 是否超越 WGDT。

---

## 12. 代码修改计划

以下修改计划以 **先做 V1 极简重构版** 为目标。

### 12.1 新增文件

#### `model/WGDT_MPBC.py`

新增新模型主文件，建议内容包括：

- `class Model(nn.Module)`
- `pre_train_step`
- `train_step`
- `test_step`
- `prediction_step`
- 多原型相关状态
- 边界对比损失计算
- consistency 计算

这是 V1 的核心文件。

#### `model/MultiPrototype.py`

新增多原型记忆模块，建议封装：

- prototype 初始化
- source 更新
- target 高置信更新
- 最近子原型查询
- prototype alignment loss
- class-wise nearest prototype 查询

---

### 12.2 修改文件

#### `main.py`

增加模型入口：

- 支持 `--model_name WGDT_MPBC`

#### `utils/meter.py`

若需要，复用当前开放集评估逻辑；
一般不需大改，仅确认兼容。

#### `utils/Trainer.py`

大概率可直接复用；
只需确认单优化器 / 多优化器流程和新模型兼容。

#### `model/__init__` 或当前模型选择逻辑相关文件

如果存在统一模型工厂，需要注册新模型。

---

## 13. 具体实现顺序

建议严格按以下顺序实现，避免一次性改太多导致无法定位问题。

### 第一步：搭建 `WGDT_MPBC` 骨架

- 复制 `WGDT.py` 为新文件 `WGDT_MPBC.py`
- 保持原始 WGDT 可跑通
- 先不加任何新损失，只确认新入口正常

### 第二步：加入 `MultiPrototype` 模块

- 先只让 source 样本更新多原型
- 加入 `L_proto_src`
- 验证 source 训练不会崩

### 第三步：加入 target 高置信原型对齐

- 根据 anchor/gamma 选 `pseudo-known`
- 加 `L_proto_tgt`
- 先不开 boundary contrastive

### 第四步：加入 `Boundary Contrastive`

- 构造 known-like / unknown-like target 样本
- 加 ranking / contrastive 边界损失
- 观察 HOS 与 unknown 变化

### 第五步：加入 `Consistency`

- 让 anchor 预测与 prototype 预测一致
- 看 known 类极端低准确率是否缓解

---

## 14. 风险与注意事项

### 14.1 主要风险

- 多原型如果更新太快，可能吸入错误 target 样本。
- 边界对比如果样本选择不准，可能导致 wrong separation。
- 如果 target 高置信筛选太松，known/unknown 会互相污染。

### 14.2 控制原则

- 第一版 target prototype 更新必须非常保守。
- 边界对比先用最简单、最稳的 ranking loss，不要一开始就上复杂 memory bank。
- 先做结构正确性验证，再做超参细调。

---

## 15. 预期收益

如果该方案有效，预期改善方向应表现为：

- `oa_known` 显著高于当前复杂 `UOT_OSDA` 线；
- 之前极低的 known 类准确率明显抬升；
- `unknown` 不至于明显崩掉；
- `HOS` 更稳定，并有机会超过 `WGDT baseline`。

---

## 16. 当前执行决策

当前正式决策如下：

1. **停止继续主修当前 `UOT_OSDA` 复杂主线。**
2. **新主线切换为 `WGDT_MPBC-v1`。**
3. **先实现极简重构版：WGDT + Multi-Prototype + Boundary Contrastive。**
4. **第一版不引入 UOT / barycenter / DANN / learnable radius。**

---

## 17. 后续对照清单

后续每次修改后，建议在本文件基础上追加记录：

- 修改日期
- 修改文件
- 修改内容
- 对应实验命令
- 关键结果
- 下一步判断

建议后续在本文件末尾新增如下格式的实验日志区：

```markdown
## 实验日志

### [YYYY-MM-DD] 实验名称
- 修改点：
- 命令：
- 结果：
- 结论：
```

这样可以保证后续实验与改动始终能一一对照。

