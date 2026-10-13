# slm-ensemble-error-correlation

Measuring teacher-student error correlation, and what it does to the
effective size of a small language model ensemble.

**Research question.** Distilled students inherit their teacher's style,
confidence, omissions, refusals and weaknesses. That inheritance is usually
described in words. This project measures it as a number: the correlation
between member errors, and the effective sample size (ESS) it implies for an
ensemble of those members.

The headline quantity is

```
ESS = N / (1 + (N - 1) * rho_bar)
```

where `rho_bar` is the mean pairwise error correlation across members. ESS is
the number of genuinely independent models your ensemble is worth. If you
train eight students from one teacher and ESS comes out near 2, you paid for
eight models and received two.

## Setup

```bash
git clone <your-fork-url> slm-ensemble-error-correlation
cd slm-ensemble-error-correlation

python -m venv .venv && source .venv/bin/activate     # Python 3.10 to 3.12
pip install --upgrade pip

# Install torch first, matched to your CUDA version, from https://pytorch.org
pip install torch

pip install -r requirements.txt
```

Datasets download themselves on first use and are cached by Hugging Face. No
data files are stored in this repository.

Some base models are gated on Hugging Face (the Llama and Gemma families).
Accept the licence on the model page, then `huggingface-cli login`. If you
would rather not deal with gating on day one, set every `student_bases` entry
in `configs/default.yaml` to a Qwen model and move on.

## The first three things to run

**1. Prove the analysis works, with no GPU and no downloads.**

```bash
python scripts/smoke_test.py
```

This fabricates traces for a teacher, a set of correlated students and an
uncorrelated control arm, runs the real estimators over them, and checks that
the pipeline recovers the structure it was handed. It should end with
`SMOKE TEST: PASS` in a few seconds. Run it again any time you change
`src/correlation.py`.

**2. Trace two models you did not train, and look at their correlation.**

```bash
python -m src.evaluate --only controls --limit 100
python -m src.analyze
```

You now have the full measurement loop running end to end on real models. The
numbers are not yet the experiment, but every later step is this loop with
more members.

**3. Distil your first student and watch the correlation appear.**

```bash
python -m src.train --recipe "T1,hard_label,0,qwen05b"
python -m src.evaluate --only students
python -m src.analyze
```

Open `runs/default/analysis/table2_inherit.csv` and read the `error_lift`
column. A value near 1.0 means your student's mistakes are unrelated to its
teacher's. A value well above 1.0 is the paper's central phenomenon, showing
up for the first time.

## Full pipeline

```bash
./scripts/run_all.sh            # train the grid, trace everything, analyse
```

or step by step:

```bash
python -m src.train --dry-run              # print the training plan
python -m src.train                        # distil the active recipe grid
python -m src.evaluate                     # trace teachers, students, controls
python -m src.analyze                      # produce the paper's tables
python -m src.plots --analysis runs/default/analysis --out figures/
```

## Layout

```
configs/default.yaml      every knob: teachers, students, recipe grid, probes
src/config.py             config loading and seeding
src/probes.py             probe suite: MCQ tasks plus refusal and omission probes
src/traces.py             run one model over the probes, record a behavioural trace
src/distill.py            distil one student under one recipe (3 objectives)
src/correlation.py        phi, Yule Q, double fault, kappa, ESS, bootstrap CIs
src/ensemble.py           vote rules, accuracy-versus-size curves, oracle bounds
src/train.py              entry point: build the student factory
src/evaluate.py           entry point: collect traces
src/analyze.py            entry point: traces in, paper tables out
src/plots.py              regenerate Figure 2 from real results
scripts/smoke_test.py     end-to-end check with synthetic traces, no GPU
experiments/log_template.md   one row per run, columns match the paper's tables
```

## How the pieces map to the paper

| Analysis output | Paper |
| --- | --- |
| `table1_main.csv` | Table 1, main result |
| `table2_inherit.csv` | Table 2, teacher-student inheritance |
| `table3_axes.csv` | Table 3, which recipe axis buys independence |
| `table4_channels.csv` | Table 4, correlation per behavioural channel |
| `scaling.csv` | Figure 2, panel (b) |

## Three things that will bite you

1. **Never train on your probe items.** `src/train.py` draws distillation
   prompts from the training splits and the probes come from test splits. If
   you blur that line you will manufacture exactly the correlation you are
   trying to measure, and the result will be meaningless.
2. **Parse failures look exactly like correlated errors.** That is why
   multiple-choice answers are read off the next-token distribution over
   option letters instead of parsed out of generated text. If you switch to
   free generation, count your parse failures and report them.
3. **Bootstrap items, not models.** The members are the fixed population you
   are studying; the probe items are your sample. `diversity_report` already
   does this, so do not "fix" it.

## Hardware

The default grid (eight students at 0.5B with LoRA) fits a single 16 GB GPU.
Tracing is cheaper than training. If you only have Colab, shrink
`active_recipes` to four rows and `max_items_per_task` to 200: the
measurement still works, the confidence intervals are just wider.
