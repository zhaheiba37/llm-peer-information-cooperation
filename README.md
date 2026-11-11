# 同伴观点与行为对LLM合作决策影响的实验

在重复公共品博弈中，独立设置同伴的实际贡献与本轮应然观点，考察二者对大语言模型智能体合作决策的影响。项目包含主实验、提示词等义改写实验和2∶1混合伙伴实验，以及提示词、理解检查、模型配置和分析程序。

## 实验设置

四名参与者共同进行10轮游戏，目标LLM决定自己的贡献，其余三名伙伴采用固定策略。每轮每人获得10积分，公共账户中的贡献总额翻倍后由四人均分。

| 实验 | 条件设置 | 每模型规模 |
|---|---|---|
| 主实验 | 伙伴均贡献10或0 × 主张贡献10、无观点或主张贡献0 | 6条件 × 50次独立游戏 × 10轮 |
| 提示词等义改写 | 两种伙伴策略 × 合作观点或无观点，使用改写模板 | 4条件 × 50次独立游戏 × 10轮 |
| 混合伙伴 | 伙伴贡献10/10/0或10/0/0 × 合作观点或无观点 | 4条件 × 50次独立游戏 × 10轮 |

| 模型 | API标识 | 推理与采样设置 |
|---|---|---|
| Qwen3.8-Flash | `qwen3.8-flash` | 开启思考，预算1024 token；temperature=0.7，top-p=0.95 |
| DeepSeek-V4.1-Flash | `deepseek-flash` | 开启思考，reasoning_effort=low；top-p=0.95 |
| GPT-6-Luna | `gpt-6-luna` | reasoning_effort=low |

输出上限均为8192 token。具体参数见`src`中的配置文件和API模块。

## 1 安装与检查

使用Python 3.11或更新版本，在项目根目录运行：

```sh
python -m pip install -r requirements.txt
python offline_check.py
```

离线检查使用模拟响应，检查收益计算、十轮交互、信息呈现及输出处理，无需API密钥。

## 2 配置API密钥

通过环境变量设置所需模型的密钥。PowerShell示例：

```powershell
$env:DASHSCOPE_API_KEY = "自己的Qwen密钥"
$env:DEEPSEEK_API_KEY = "自己的DeepSeek密钥"
$env:OPENAI_API_KEY = "自己的OpenAI密钥"
```

运行前，在配置文件和执行脚本中设置费用估计与本地预算上限。API实验会产生调用费用。

## 3 生成实验计划

根据需要选择一项：

```sh
python reproduce.py --study main
python reproduce.py --study main-gpt
python reproduce.py --study paraphrase
python reproduce.py --study mixed
```

| `--study` | 内容 |
|---|---|
| `main` | DeepSeek和Qwen主实验，依次执行 |
| `main-gpt` | GPT主实验 |
| `paraphrase` | 三模型共用的等义改写实验计划 |
| `mixed` | 三模型共用的混合伙伴实验计划 |

生成计划不会调用API。命令将打印`PREPARED`目录，该目录位于`src/runs/`，用于下一步的`--run`参数。

## 4 执行实验

将以下路径替换为对应的`PREPARED`目录，并添加`--execute`开始采集：

```sh
python reproduce.py --study main --execute --run "src/runs/对应main目录"
python reproduce.py --study main-gpt --execute --run "src/runs/对应main-gpt目录"
python reproduce.py --study paraphrase --execute --run "src/runs/对应paraphrase目录" --provider qwen
python reproduce.py --study mixed --execute --run "src/runs/对应mixed目录" --provider deepseek
```

两项扩展实验通过`--provider`指定模型，可选`qwen`、`deepseek`或`gpt`；在同一计划下分别执行三次即可完成三模型采集。

程序先进行20项理解检查，全部通过后开始行为实验。技术性未完成游戏由同条件的新游戏补充，直至每条件获得50次完整游戏。如需重新开展整批实验，先重新生成计划。

## 5 查看与分析结果

执行程序会打印结果目录，其中主要文件为：

| 文件 | 内容 |
|---|---|
| `plan.json` | 实验配置和任务安排 |
| `raw.jsonl` | API请求与响应记录 |
| `decisions.jsonl` | 各轮贡献和收益 |
| `episodes.jsonl` | 游戏状态及完整贡献历史 |
| `qualification/` | 理解检查结果 |
| `complete50_view/` | 每条件50次完整游戏的结果视图 |

对包含`episodes.jsonl`的模型结果目录运行分析：

```sh
python analyze.py "src/runs/某次运行/qwen" --output analysis_output/qwen
```

将示例路径替换为程序实际输出的模型结果目录。主实验GPT结果直接位于其`PREPARED`目录；其余实验的模型结果位于相应的模型子目录。分析输入可使用原结果目录或其`complete50_view`，选择其中一个即可。

程序输出平均贡献、合作观点相对无观点的贡献差、95%置信区间及轨迹图。置信区间使用5,000次完整游戏重采样，混合伙伴按少数伙伴身份分层。

仅计算统计结果时可使用：

```sh
python analyze.py "src/runs/某次运行/qwen" --output analysis_output/qwen --no-plot
```

## 主要代码

| 文件 | 功能 |
|---|---|
| `reproduce.py` | 统一实验入口 |
| `src/run_formal_v11.py`、`src/run_gpt_luna_formal.py` | 主实验采集 |
| `src/run_from_round1.py` | 逐轮交互与历史反馈 |
| `src/prompts_v11_public_observation.py` | 主实验规则 |
| `src/prompts_paraphrase_b.py`、`src/run_paraphrase_b.py` | 等义提示词与实验采集 |
| `src/mixed_partner_core.py`、`src/run_mixed_partner.py` | 混合伙伴设置与实验采集 |
| `src/run_public_observation_pilot.py`中的`tasks()` | 20项理解检查 |
| `src/qwen_api.py`、`src/deepseek_api.py`、`src/gpt_luna_recovery_api.py` | 模型API调用 |
| `offline_check.py` | 离线流程检查 |
| `analyze.py` | 统计分析与绘图 |

提示词由上述模块组装。各轮提供此前全部真实贡献和目标智能体自身收益，以及本轮伙伴观点；第1轮没有行为历史。
