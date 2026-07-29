# Contributing

This repository treats numerical evidence as part of the code.

Before opening a change:

```bash
uv sync --extra dev
uv run ruff check .
uv run pytest
```

If a change affects a checked result, run its validator under `scripts/` and
regenerate the artifact from a clean source commit. Keep the manifest's source
revision, data hash, model hash, and claim boundary together.

For scientific changes:

- state the coefficient and coordinate conventions in code;
- split complete trajectories rather than individual time rows;
- keep oracle quantities out of training;
- include the closest independent baseline and a negative control;
- predeclare thresholds before promoting a result;
- preserve failed controls and abstentions in the artifact; and
- distinguish structural identities, current-dataset evidence, numerical
  stability, and formal guarantees.

Bug fixes, clearer derivations, additional physical systems, fair baseline
implementations, and attempts to falsify the promoted results are especially
welcome.
