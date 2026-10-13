"""Error correlation and the real value of SLM ensembles.

Package layout:

    config.py       load and validate configs/default.yaml
    probes.py       build the probe suite (MCQ tasks plus refusal and omission probes)
    traces.py       run one model over the probe suite and record a behavioural trace
    distill.py      distil a student from a teacher under one recipe
    correlation.py  pairwise error-correlation and effective-sample-size estimators
    ensemble.py     vote rules and accuracy-versus-size curves
    train.py        entry point: build the student factory
    evaluate.py     entry point: collect traces for every member
    analyze.py      entry point: traces in, paper tables and figures out
"""

__version__ = "0.1.0"
