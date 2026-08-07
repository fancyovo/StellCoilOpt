# Contributing

## Development setup

```bash
python -m pip install -e ".[plot,dev]"
python -m compileall -q .
python -m pytest -q
```

Changes to the native ABI must update the C header, CUDA implementation and Python `ctypes` structures together. Increment the ABI version when structure layout or function contracts change.

## Numerical changes

A numerical change should include a focused test that states the invariant being protected. Keep the fast evaluator bounded: new iterative stages need explicit iteration caps, and failure must return a status rather than silently changing algorithms or using a slower fallback.

Do not commit datasets, model weights, generated run directories, benchmark output or machine-specific launch files. Examples must be small, redistributable and free of private provenance.

## Documentation

Method documentation should define symbols before use and distinguish the fast screening score from full physical acceptance. Do not add performance or quality comparisons without a self-contained experiment definition and publishable artifacts.
