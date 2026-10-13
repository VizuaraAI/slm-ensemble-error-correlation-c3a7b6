# Experiment log

One row per run. The columns are the columns of the paper's tables, so a
finished row can be copied into the draft without re-deriving anything.

Rules that keep this log worth having:

- Write the row **before** you launch the run, with the hypothesis filled in.
  A hypothesis written afterwards is a description, not a prediction.
- Never edit a row after its run finished. Add a new row instead.
- `run_id` is the `output_dir` of the run. Keep them unique and keep the
  traces: the whole point of the trace format is that you never need to
  reload a model to re-analyse it.
- If a run fails or is abandoned, keep the row and write why in Notes. The
  abandoned runs are where the methods section comes from.

---

## Table 1: main result (one row per ensemble construction strategy)

Maps to Table 1 of the paper and to `analysis/table1_main.csv`.

| run_id | date | strategy | N | member acc % | vote acc % | oracle acc % | rho_bar | ESS | ESS 95% CI | N/ESS | all-wrong % | saturation N | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | | Single model (reference) | 1 | | | | | | | | | | |
| | | Self-consistency, one student | | | | | | | | | | | |
| | | Same teacher, seed only | | | | | | | | | | | |
| | | Same teacher, KD varied | | | | | | | | | | | |
| | | Same teacher, base varied | | | | | | | | | | | |
| | | Two teachers, mixed recipes | | | | | | | | | | | |
| | | Independent SLMs (control) | | | | | | | | | | | |

## Table 2: teacher-student inheritance (one row per distillation objective)

Maps to Table 2 and to `analysis/table2_inherit.csv`.

| run_id | objective | n students | rho_TS | P(S wrong given T wrong) | P(S wrong given T right) | error lift | inherited share % | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | hard_label | | | | | | | |
| | seq_kd | | | | | | | |
| | reverse_kl | | | | | | | |
| | different teacher (cross check) | | | | | | | |
| | no teacher (control) | | | | | | | |

## Table 3: which recipe axis buys independence

Maps to Table 3 and to `analysis/table3_axes.csv`. Fill `gpu hours` by hand
from your training logs; nothing else can measure it for you.

| run_id | axis varied | rho within | rho between | delta rho | delta ESS at N=8 | gpu hours | independence per gpu hour | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | random seed only | | | | | | | |
| | data shard | | | | | | | |
| | distillation objective | | | | | | | |
| | student base family | | | | | | | |
| | teacher identity | | | | | | | |

## Table 4: correlation per behavioural channel

Maps to Table 4 and to `analysis/table4_channels.csv`.

| run_id | channel | n items | rho_bar | ESS | ESS 95% CI | covariance share % | mean error rate | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | correctness (mcq) | | | | | | | |
| | confidence (binned) | | | | | | | |
| | refusal | | | | | | | |
| | omission | | | | | | | |

---

## Run narrative

Copy this block per run. Three sentences each is plenty.

### run_id: `runs/____`

- **Date:**
- **Config diff from default:**
- **Hypothesis (written before launching):**
- **What actually happened:**
- **Does it change the paper?** (which section, which table, which sentence)
- **Next run this implies:**

---

## Example of a completed entry

### run_id: `runs/2026-10-20-controls-only`

- **Date:** 2026-10-20
- **Config diff from default:** `active_recipes` empty; traced the three
  control models only; `max_items_per_task: 200`.
- **Hypothesis (written before launching):** Independently pretrained SLMs of
  this size should still correlate somewhat, because they share pretraining
  data and benchmark contamination, but `rho_bar` should land well below 0.3.
  This is the reference point for every distilled ensemble.
- **What actually happened:** _fill in._
- **Does it change the paper?** Sets the control row of Table 1 and the upper
  anchor in Figure 2, panel (a).
- **Next run this implies:** One student per objective from teacher T1, to see
  whether `error_lift` separates the three objectives before scaling the grid.
