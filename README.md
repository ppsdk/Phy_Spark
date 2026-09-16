# Phy_Spark / PhysGround-Tune

[![CI](https://github.com/ppsdk/Phy_Spark/actions/workflows/ci.yml/badge.svg)](https://github.com/ppsdk/Phy_Spark/actions/workflows/ci.yml)

**PhysGround-Tune** 是一个面向视觉语言模型（VLM）的物理状态表征后训练代码库。目标不是重新搭建物理模拟器，而是复用 CLEVRER、Physion++ 与公开 PhysInOne 训练数据中已经存在、可核验的物理真值，把 **object state、latent physical property、physical relation、multi-horizon state change** 直接监督到 VLM 的中间表示，并与普通物理 QA LoRA SFT 做公平比较。

首版代码围绕以下研究问题实现：

> 在相同 backbone、训练媒体、LoRA 预算和训练步数下，显式 physical representation grounding 是否比“只把同样的标签写成文本做 SFT”更有利于跨模型、跨规模、跨 benchmark 的物理推理迁移？

当前版本不依赖独立模拟器，不运行物理引擎，不重新渲染训练视频，也不自动制造“违反物理规律”的负样本。

## 1. 已实现内容

- Hugging Face `Transformers + PEFT + Trainer` 的统一 VLM LoRA 训练入口。
- Qwen3-VL 与 InternVL3.5 **HF-format** 共用的 multimodal chat-template 数据通路。
- 普通 QA / 物理 QA 的 LoRA SFT baseline。
- 从“答案生成前”的 prompt-end hidden state 读取表示，避免辅助物理头直接使用答案 token。
- 可配置的 state / property / relation / constraint / delta 分类或回归头。
- `delta` 双观察时刻编码：分别编码截至 `t` 与 `t+τ` 的可见内容，再监督已观察到的状态变化。
- 可选 V-JEPA 动态差分蒸馏接口：训练样本提供预计算的 `teacher_delta` 后即可启用 cosine loss；教师本身不在推理阶段使用。
- 按字段 mask 的监督机制：数据源没有可靠标签时，不自动补标签，也不把缺失字段当负例。
- PhysBench 官方 `test.json` 直接评测。
- Physion++ 严格按 `start_frame_for_prediction` 截断视频前缀的直接 QA 诊断。
- Physion++ 冻结表示 + Logistic Regression readout 评测，用于更接近官方 representation/readout protocol 的比较。
- IntPhys2 Debug/Main split 的 possible / impossible VLM 评测。

## 2. 研究框架

训练期模型输入仍然是视觉内容和文本问题：

```text
Video / Images + Question
        ↓
       VLM
        ↓
  prompt-end h_t
   ↙    ↓     ↘
state property delta heads
        ↓
  physical grounding losses
```

最终部署接口保持不变：

```text
Video + Question → VLM → Answer
```

辅助物理头和 V-JEPA teacher 只用于训练/分析，普通 benchmark 推理只需要 base VLM + LoRA adapter。

总损失可写为：

```text
L = λ_lm L_lm
  + Σ_a w_a L_a(state/property/relation/constraint)
  + Σ_d w_d L_d(delta)
  + λ_jepa L_jepa
```

其中每个字段独立 mask。首版默认 `constraint` 与 `jepa` 都关闭；只有当训练源存在经过核验的标签/teacher feature 时再显式启用。

## 3. 安装

建议 Python 3.10+、CUDA 环境和较新的 PyTorch。

```bash
git clone https://github.com/ppsdk/Phy_Spark.git
cd Phy_Spark

python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

Qwen3-VL 需要较新的 Transformers；本项目依赖写为 `transformers>=4.57,<6`。如果某个最新 checkpoint 明确要求更新版本，请优先遵循该 checkpoint 的模型卡。

## 4. 目录结构

```text
Phy_Spark/
├── README.md
├── requirements.txt
├── pyproject.toml
├── train.py
├── configs/
│   ├── qwen3vl_sft.yaml
│   ├── qwen3vl_physground.yaml
│   └── internvl35_physground.yaml
├── physground/
│   ├── config.py
│   ├── data.py
│   ├── inference.py
│   ├── media.py
│   ├── modeling.py
│   ├── trainer.py
│   └── utils.py
├── eval/
│   ├── physbench.py
│   ├── physionpp.py
│   ├── physionpp_readout.py
│   └── intphys2.py
├── scripts/
│   ├── build_physionpp_manifest.py
│   └── validate_manifest.py
└── examples/
    └── train_example.jsonl
```

## 5. 训练数据：统一 canonical JSONL

不同数据集的原始字段差异很大，所以训练核心不直接猜测某个版本的 raw schema，而是统一读取 canonical JSONL。**数据集适配层负责把“已发布或可靠派生的标签”转换到这个格式；训练代码只消费已经核验过的字段。**

一个样本示例：

```json
{
  "id": "sample-0001",
  "source": "clevrer",
  "content": [
    {"type": "video", "path": "clevrer/video_0001.mp4"},
    {"type": "text", "text": "What happens to the red object?"}
  ],
  "answer": "It collides with the blue object.",
  "targets": {
    "moving": 1,
    "contact": 1,
    "mass_rank": 2
  },
  "target_masks": {
    "moving": 1,
    "contact": 1,
    "mass_rank": 0
  },
  "delta": {
    "tau": 0.5,
    "t": {
      "media": [{"type": "video", "path": "clips/0001_t.mp4"}]
    },
    "tp": {
      "media": [{"type": "video", "path": "clips/0001_tp.mp4"}]
    }
  }
}
```

提交训练任务前可先做不加载模型权重的数据预检：

```bash
python scripts/validate_manifest.py examples/train_example.jsonl \
  --config configs/qwen3vl_physground.yaml

# 提供 media-root 时还会检查本地媒体和 teacher cache 是否存在
python scripts/validate_manifest.py data/train_physground.jsonl \
  --config configs/qwen3vl_physground.yaml \
  --media-root data
```

预检会核对 head 名称、分类标签范围、回归向量维度、mask、delta 两端、V-JEPA teacher 维度及媒体路径。`train.py` 在下载或加载模型权重前也会执行同一检查。

### 5.1 字段语义

| 字段 | 说明 |
|---|---|
| `content` | 保持图像/视频/文本的真实顺序；兼容 Hugging Face multimodal chat format |
| `answer` | 普通 SFT 文本答案；纯 representation sample 可省略 |
| `targets` | 辅助监督标签；key 必须对应 config 中的 head 名称 |
| `target_masks` | 可选。`0` 表示标签不可靠或不适用；缺失 target 默认不计算 loss |
| `delta.t` / `delta.tp` | 两个**独立可见观察**。不要先编码完整未来视频再切 hidden token |
| `delta.tau` | 时间跨度，建议使用秒或数据集内统一时间单位 |
| `teacher_delta` | 可选，预计算 V-JEPA dynamic feature difference 向量 |
| `teacher_delta_path` | 可选，保存上述向量的 `.npy` 路径 |

如果视频片段不想预先切文件，media item 支持：

```json
{"type":"video", "path":"trial.mp4", "start_frame":0, "end_frame":120, "num_frames":16}
```

训练时会只解码这个区间。

### 5.2 不要做的标签转换

- CLEVRER 的颜色、材质名称不能直接改写成 mass / friction 真值。
- collision 不能直接当 support。
- 没有标注的 relation 不能默认设为 0。
- 只有模拟器参数、但输入视频没有足够辨识线索时，不应强行监督绝对属性值。
- 没有可靠 valid/invalid 视频标签时，不开启 constraint classification。

## 6. Head 配置

`configs/qwen3vl_physground.yaml` 中每个 head 都显式声明：

```yaml
grounding:
  lm_weight: 1.0
  jepa_weight: 0.0
  heads:
    - name: moving
      group: state
      kind: classification
      num_labels: 2
      weight: 1.0

    - name: restitution
      group: property
      kind: regression
      num_labels: 1
      weight: 0.5

    - name: contact_onset
      group: delta
      kind: classification
      num_labels: 2
      weight: 1.0
```

支持：

- `kind: classification`
- `kind: regression`
- `kind: bce`

`group: delta` 的 head 自动读取 `h_{t+τ} - h_t` 与 `τ` 融合后的表示；其他 group 默认读取主问题生成答案前的表示。

## 7. Baseline：VLM-PEFT-SFT

同一训练代码可以直接跑只有文本答案的 LoRA SFT baseline：

```bash
python train.py --config configs/qwen3vl_sft.yaml
```

SFT manifest 可以完全不包含 `targets` 和 `delta`。这样 Base / SFT / PhysGround-Tune 能复用同一 processor、抽帧、LoRA 和 Trainer 路径。

## 8. PhysGround-Tune 训练

### Qwen3-VL

```bash
python train.py --config configs/qwen3vl_physground.yaml
```

### InternVL3.5

推荐使用 Hugging Face 标准格式的 checkpoint，例如：

```bash
python train.py --config configs/internvl35_physground.yaml
```

配置文件中默认示例为 `OpenGVLab/InternVL3_5-8B-HF`。非 HF-format / legacy remote-code checkpoint 的前向接口可能不同，不建议用于跨 backbone 的公平主实验。

### 有效 batch size

v0.1 为了保持不同 VLM 的 variable video-patch packing 一致，固定：

```text
per_device_train_batch_size = 1
```

需要通过：

```yaml
gradient_accumulation_steps: 8
```

提高有效 batch size。这样比在 collator 中手写 Qwen/InternVL 两套不同视觉 patch 拼接逻辑更容易保证公平。

训练输出：

```text
outputs/.../
├── adapter/                 # PEFT LoRA adapter
├── physground_heads.pt      # state/property/delta/JEPA heads
├── physground_config.json
└── processor/
```

benchmark 推理只需要 `adapter/`；辅助 heads 不参与普通 QA generation。

## 9. V-JEPA dynamic teacher

当前代码支持 **预计算 teacher difference**，不在训练进程中加载 V-JEPA：

```yaml
grounding:
  jepa_weight: 0.2
  jepa_dim: 1024
```

样本提供：

```json
{"teacher_delta": [0.01, -0.12, ...]}
```

或：

```json
{"teacher_delta_path": "jepa_cache/sample_0001.npy"}
```

代码对 VLM 两端状态差分做 projection，再用 cosine distance 对齐 teacher difference。teacher target stop-gradient；推理时完全移除。

## 10. PhysBench 评测

官方 PhysBench `test.json` 通过 `<video>` / `<image>` 占位符定义视觉输入顺序。评测代码按照 `file_name` 顺序恢复 multimodal content，并强制模型最终输出 A/B/C/D。

数据目录示例：

```text
/path/to/PhysBench/
├── test.json
├── image/
└── video/
```

运行 Base VLM：

```bash
python eval/physbench.py \
  --model Qwen/Qwen3-VL-4B-Instruct \
  --data-root /path/to/PhysBench \
  --output results/physbench_base.jsonl
```

运行 PhysGround adapter：

```bash
python eval/physbench.py \
  --model Qwen/Qwen3-VL-4B-Instruct \
  --adapter outputs/qwen3vl-4b-physground \
  --data-root /path/to/PhysBench \
  --output results/physbench_physground.jsonl
```

输出除 Overall accuracy 外，还会按当前记录中存在的 `task_type / sub_type / ability_type / source / split` 自动分组。

## 11. Physion++

Physion++ 官方 release 的 `.pkl` metadata 包含场景物理参数、对象轨迹、collision events、trial label 和 `start_frame_for_prediction`。本项目严格使用：

```text
visible input = frames [0, start_frame_for_prediction)
```

而不会把未来 prediction phase 图像送进模型。

### 11.1 生成 manifest

对官方可信 release 运行：

```bash
python scripts/build_physionpp_manifest.py \
  --root /path/to/Readout_data \
  --output data/physionpp_readout.jsonl

python scripts/build_physionpp_manifest.py \
  --root /path/to/Testing_data \
  --output data/physionpp_test.jsonl
```

> 该脚本会读取 pickle。只对官方、可信的 Physion++ 数据文件运行，不要对未知来源 pickle 使用。

如果某个 release 版本里的 label key 与常见字段不同，脚本会保留可解析记录并报告 unresolved label，而不是猜测答案。

### 11.2 直接 VLM Yes/No 诊断

```bash
python eval/physionpp.py \
  --model Qwen/Qwen3-VL-4B-Instruct \
  --adapter outputs/qwen3vl-4b-physground \
  --manifest data/physionpp_test.jsonl \
  --data-root /path/to/Testing_data \
  --output results/physionpp_qa.jsonl
```

这是 **direct QA diagnostic**，便于和普通 VLM 统一接口比较；它不应与论文中的 frozen representation readout protocol 混为一个数字。

### 11.3 冻结表示 + readout

```bash
python eval/physionpp_readout.py \
  --model Qwen/Qwen3-VL-4B-Instruct \
  --adapter outputs/qwen3vl-4b-physground \
  --readout-manifest data/physionpp_readout.jsonl \
  --readout-root /path/to/Readout_data \
  --test-manifest data/physionpp_test.jsonl \
  --test-root /path/to/Testing_data \
  --output results/physionpp_readout.jsonl
```

该脚本冻结 VLM，提取 prediction point 前的 prompt-end representation，然后按 property 单独训练 Logistic Regression，并在对应测试 property 上报告 accuracy。

由于主研究方案会使用 Physion++ dynamics training，因此主结果应表述为 **official split generalization**；只有做“排除 Physion++ dynamics train”的独立实验时，才能把 Physion++ 结果称为未见数据源的 OOD transfer。

## 12. IntPhys2

IntPhys2 作为 constraint / intuitive-physics transfer benchmark 使用，不进入首版训练集。

官方目录示例：

```text
/path/to/IntPhys2/
├── Debug/
│   ├── metadata.csv
│   └── Videos/
└── Test/
    ├── metadata.csv
    └── Videos/
```

运行：

```bash
python eval/intphys2.py \
  --model Qwen/Qwen3-VL-4B-Instruct \
  --adapter outputs/qwen3vl-4b-physground \
  --data-root /path/to/IntPhys2 \
  --split Test \
  --output results/intphys2.jsonl
```

当前 evaluator 对每个视频**独立**判断 Yes/No，不把 possible/impossible 配对答案作为输入，并同时报告单视频 accuracy 与 scene-level all-correct 指标。Held-Out split 官方没有公开 ground-truth metadata，本脚本不伪造其标签。

请遵守 IntPhys2 数据许可：其官方仓库将该数据限定为评测用途，并对其他使用方式有额外限制。

## 13. Representation-level verification

论文主实验建议固定相同 probe protocol，比较：

```text
Base VLM
VLM-PEFT-SFT
Same-label textual SFT
PhysGround-Tune
```

probe 目标可包括：

```text
motion / contact / support
mass / friction / restitution
Δvelocity / contact_onset / contact_offset
```

核心证据链应是：

```text
Physical Grounding
        ↓
Physical State Identifiability
        ↓
Cross-benchmark Physical Reasoning
```

仅 probe 变好还不足以证明“更好的物理推理”；需要同时看 PhysBench、Physion++、IntPhys2 的 downstream 结果。

## 14. 推荐实验矩阵

至少：

```text
Backbone families:
  Qwen3-VL
  InternVL3.5-HF

Scales:
  2B / 4B / 8B（按实际公开 checkpoint 对齐）

Methods:
  Base
  VLM-PEFT-SFT
  Same-label Text SFT
  + State
  + Property
  + Delta
  + V-JEPA
  Full

Benchmarks:
  PhysBench
  Physion++ direct QA
  Physion++ readout
  IntPhys2
```

公平性必须固定：训练媒体、抽帧策略、LoRA rank/target modules、优化器、训练 token 数/steps、prompt 与 benchmark parser。

## 15. 当前边界

1. **本仓库不自行推断 raw dataset 的物理标签。** CLEVRER / Physion++ / PhysInOne 的 raw → canonical 转换应以你实际下载版本的字段为准；无法核验的字段必须 mask。
2. `support`、`deformation`、绝对 mass/friction 数值等是否启用，取决于具体数据源的真实字段和可辨识性，不因 config 中存在 head 就自动生成标签。
3. Physion++ `build_physionpp_manifest.py` 是面向官方 release 的便捷扫描器；不同 release 如果 metadata key 有变化，应显式适配，不要通过 LLM 猜测。
4. 目前训练 collator 使用 batch size 1 + gradient accumulation；这是为了先保证 Qwen3-VL / InternVL 的视频输入一致性和可复现性。
5. V-JEPA teacher 目前采用离线 cache；这是刻意设计，避免让训练进程耦合第二个大型视频模型，也便于精确统计额外训练成本。

## 16. 数据与实现依据

- CLEVRER: released train videos/questions/scene/event annotations
- Physion++: dynamics training / readout fitting / testing split，以及 `start_frame_for_prediction`
- PhysBench: image-video-text interleaved multiple-choice benchmark
- IntPhys2: permanence / immutability / spatio-temporal continuity / solidity violation-of-expectation benchmark
- Hugging Face Transformers multimodal chat templates
- PEFT LoRA
- Qwen3-VL
- InternVL3.5 HF-format

本仓库的目标是让**算法定义、数据标签边界和 benchmark protocol 分开**：模型可以换、数据源可以换，但不能为了“跑通”而把不确定的物理字段偷偷变成监督真值。
