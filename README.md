# Experiments on the Effects of Peer Opinions and Behavior on LLM Cooperation Decisions

English | [简体中文](README_zh.md)

This project examines how peer opinions and observed behavior affect cooperation decisions of large language model (LLM) agents in repeated public goods games. It includes the main experiment, a prompt-paraphrasing experiment, and a mixed-partner experiment, together with prompts, comprehension checks, model configurations, and analysis scripts.

## Experimental setup

Four participants play ten rounds. The target LLM chooses its contribution, while three scripted partners follow fixed contribution strategies. In each round, every participant receives ten points. Contributions to the common account are doubled and divided equally among all four participants.

| Experiment | Conditions | Design per model |
|---|---|---|
| Main experiment | All partners contribute 10 or 0 × opinions advocating 10, no opinion, or opinions advocating 0 | 6 conditions × 50 independent games × 10 rounds |
| Prompt paraphrasing | Two partner strategies × cooperative opinions or no opinion, using paraphrased prompts | 4 conditions × 50 independent games × 10 rounds |
| Mixed partners | Partner contributions of 10/10/0 or 10/0/0 × cooperative opinions or no opinion | 4 conditions × 50 independent games × 10 rounds |

| Model | API identifier | Reasoning and sampling settings |
|---|---|---|
| Qwen3.8-Flash | `qwen3.8-flash` | Thinking enabled; budget: 1,024 tokens; temperature=0.7; top-p=0.95 |
| DeepSeek-V4.1-Flash | `deepseek-flash` | Thinking enabled; reasoning_effort=low; top-p=0.95 |
| GPT-6-Luna | `gpt-6-luna` | reasoning_effort=low |

The output token limit is 8,192 for all three models. See the configuration files and API modules in `src` for details.

## 1. Installation and offline checks

Use Python 3.11 or later. From the project root, run:

```sh
python -m pip install -r requirements.txt
python offline_check.py
```

The offline checks use simulated responses to verify payoff calculations, ten-round interactions, information presentation, and output handling. No API key is required.

## 2. Configure API keys

Set the environment variables for the providers you will use. For example, in PowerShell:

```powershell
$env:DASHSCOPE_API_KEY = "your-qwen-api-key"
$env:DEEPSEEK_API_KEY = "your-deepseek-api-key"
$env:OPENAI_API_KEY = "your-openai-api-key"
```

Before running experiments, configure the cost estimates and local spending limits in the configuration files and execution scripts. API execution incurs charges.

## 3. Prepare an experiment plan

Choose the study to prepare:

```sh
python reproduce.py --study main
python reproduce.py --study main-gpt
python reproduce.py --study paraphrase
python reproduce.py --study mixed
```

| `--study` | Experiment |
|---|---|
| `main` | Main experiment for DeepSeek and Qwen, executed sequentially |
| `main-gpt` | Main experiment for GPT |
| `paraphrase` | Prompt-paraphrasing plan for all three models |
| `mixed` | Mixed-partner plan for all three models |

Plan preparation makes no API calls. Each command prints a `PREPARED` directory under `src/runs/`. Use that directory as the `--run` argument in the next step.

## 4. Run the experiment

Replace each example path with the corresponding `PREPARED` directory and add `--execute` to start collection:

```sh
python reproduce.py --study main --execute --run "src/runs/your-main-plan"
python reproduce.py --study main-gpt --execute --run "src/runs/your-main-gpt-plan"
python reproduce.py --study paraphrase --execute --run "src/runs/your-paraphrase-plan" --provider qwen
python reproduce.py --study mixed --execute --run "src/runs/your-mixed-plan" --provider deepseek
```

For the two extension experiments, select `qwen`, `deepseek`, or `gpt` with `--provider`. Run each provider separately under the same study plan to collect results for all three models.

The program first administers 20 comprehension checks. Behavioral collection begins after all checks pass. Technically incomplete games are replaced with new games under the same condition until each condition has 50 complete games. Prepare a new plan to start a new batch.

## 5. Inspect and analyze results

The execution script prints the results directory. Its main files are:

| File or directory | Contents |
|---|---|
| `plan.json` | Experimental configuration and job schedule |
| `raw.jsonl` | API requests and responses |
| `decisions.jsonl` | Contributions and payoffs for each round |
| `episodes.jsonl` | Game status and complete contribution histories |
| `qualification/` | Comprehension-check results |
| `complete50_view/` | Results containing 50 complete games per condition |

Run the analysis on a model results directory containing `episodes.jsonl`:

```sh
python analyze.py "src/runs/your-run/qwen" --output analysis_output/qwen
```

Replace the example path with the actual output directory. Main-experiment GPT results are stored directly in its `PREPARED` directory; other runs store model results in provider subdirectories. Use either the original results directory or its `complete50_view` as the analysis input.

The analysis produces mean contributions, contribution differences between cooperative-opinion and no-opinion conditions, 95% confidence intervals, and trajectory plots. Confidence intervals use 5,000 bootstrap resamples of complete games, stratified by minority-partner identity in the mixed-partner experiment.

To compute statistics without generating plots:

```sh
python analyze.py "src/runs/your-run/qwen" --output analysis_output/qwen --no-plot
```

## Main files

| File | Purpose |
|---|---|
| `reproduce.py` | Unified experiment entry point |
| `src/run_formal_v11.py`, `src/run_gpt_luna_formal.py` | Main-experiment collection |
| `src/run_from_round1.py` | Round-by-round interactions and historical feedback |
| `src/prompts_v11_public_observation.py` | Main-experiment rules |
| `src/prompts_paraphrase_b.py`, `src/run_paraphrase_b.py` | Paraphrased prompts and experiment collection |
| `src/mixed_partner_core.py`, `src/run_mixed_partner.py` | Mixed-partner settings and collection |
| `tasks()` in `src/run_public_observation_pilot.py` | The 20 comprehension checks |
| `src/qwen_api.py`, `src/deepseek_api.py`, `src/gpt_luna_recovery_api.py` | Model API clients |
| `offline_check.py` | Offline workflow checks |
| `analyze.py` | Statistical analysis and plotting |

Prompts are assembled by these modules. Each round presents all previous actual contributions, the target agent's own payoffs, and the partners' current-round opinions. The first round contains no behavioral history.
