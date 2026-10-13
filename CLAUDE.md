# Instructions for the coding agent

You are helping Ashish Kumar Sinha run a research project for the Vizuara SLM
Bootcamp. He will work with you most days for about eight weeks. Read this
file before touching anything.

## The goal, in one paragraph

Distilled small language models inherit their teacher's mistakes. We are
measuring how much, and converting that measurement into the effective sample
size (ESS) of an ensemble of such models:

```
ESS = N / (1 + (N - 1) * rho_bar)
```

`rho_bar` is the mean pairwise error correlation across ensemble members. The
deliverable is a sizing rule: given a distillation recipe, how many students
are worth training before the next one adds compute and nothing else. The
target output is a workshop paper. The paper draft lives alongside this repo
in `paper/main.tex` and its tables are already laid out; the analysis code
emits CSVs whose columns match those tables by name.

## Repository layout

```
configs/default.yaml   every experimental knob, including the recipe grid
src/config.py          config loading, seeding, recipe naming
src/probes.py          probe suite construction (MCQ, refusal, omission)
src/traces.py          model wrapper and behavioural trace collection
src/distill.py         distillation: hard_label, seq_kd, reverse_kl
src/correlation.py     THE SCIENTIFIC CORE: phi, Yule Q, ESS, bootstrap
src/ensemble.py        vote rules, accuracy-versus-size curves
src/train.py           entry point: build students
src/evaluate.py        entry point: collect traces
src/analyze.py         entry point: traces in, paper tables out
src/plots.py           regenerate Figure 2 from real results
scripts/smoke_test.py  end-to-end check on synthetic traces, no GPU needed
experiments/log_template.md  the running record of what was tried
```

## Conventions

- **Python 3.10+, type hints on function signatures, no classes unless state
  genuinely needs to live somewhere.** `ScoredModel` earns its class; nothing
  else so far does.
- **`src/correlation.py` depends only on numpy.** Keep it that way. It is the
  part of the project that must be obviously correct, and that is much easier
  when it has no heavy dependencies and runs in milliseconds.
- **Every change to an estimator is followed by `python scripts/smoke_test.py`.**
  If you add an estimator, add a check to the smoke test asserting its value
  on an input whose answer is known by hand.
- **Traces are append-only artefacts.** Never overwrite a trace file that has
  already been written about in `experiments/log_template.md`. Write a new
  `output_dir` instead.
- **Seeds are explicit.** `set_global_seed` is called at the top of every
  entry point. Any new randomness takes an `rng` argument rather than calling
  the global numpy generator.
- **No em dashes in code comments, docstrings or paper text.** Use commas,
  colons, parentheses or semicolons.

## Statistical rules that are not negotiable

These are the places where a plausible-looking change silently invalidates the
result. Push back if asked to change them without a reason.

1. **Bootstrap over items, never over models.** The ensemble members are the
   fixed population under study. The probe items are the sample. Resampling
   models would answer a different question.
2. **Never train on probe items.** Distillation prompts come from training
   splits; probes come from test splits. Violating this manufactures the
   correlation we are measuring.
3. **Multiple-choice answers are scored from the next-token distribution over
   option letters, not parsed from generated text.** A parse failure is
   indistinguishable from a correlated error and would contaminate `rho_bar`.
4. **The control arm matters.** Independently pretrained SLMs bound how much
   diversity is available at this scale at all. Without it, a low ESS might
   just mean "these models are small", not "these models share a teacher".
5. **Report `rho_bar` with its confidence interval.** A point estimate of
   correlation from a few hundred items is noisier than it looks.

## What to do next

Current state: the pipeline runs end to end, verified on synthetic traces.
Nothing has been trained yet.

1. Run `python scripts/smoke_test.py` and confirm it passes.
2. Trace the control models (`python -m src.evaluate --only controls`) and get
   a first real `rho_bar` for models that share no teacher. This is the
   reference point everything else is compared against.
3. Distil one student per objective from teacher `T1` and check whether
   `error_lift` in `table2_inherit.csv` separates the three objectives.
4. Widen `active_recipes` to the full grid only after step 3 produces a signal
   worth scaling up. Training eight students before knowing the measurement
   works is the main way this project wastes a month.
5. Fill `gpu_hours` in `table3_axes.csv` from the training logs. The
   cost-effectiveness ranking is the part a practitioner will actually use.

## When Ashish asks for something ambiguous

Prefer the version that produces a number he can put in a table. This project
is a measurement, not a system: a change that makes the pipeline faster but
the estimate less interpretable is the wrong trade. If a request would break
one of the five statistical rules above, say so explicitly and propose the
nearest version that does not.
