# Project guidance

- Run from the repository root. Use the Python 3.12 environment at `.venv`; the system `python3` is 3.14 on the development machine.
- Install exact dependencies with `uv pip install --python .venv/bin/python -r requirements-lock.txt`. Do not update dependencies during the demo.
- Verification: `MUJOCO_GL=egl .venv/bin/python -m pytest -q`.
- Integration smoke test: `MUJOCO_GL=egl .venv/bin/python -m warehouse.demo --seed 7 --perception oracle --transport idealized --headless --auto`.
- Benchmark: `MUJOCO_GL=egl .venv/bin/python -m warehouse.evaluate --seeds 0:20 --perception oracle --transport idealized`.
- Exception scenarios: `lateral missing empty unknown_stock blocked_return pick_timeout`; pass them to `--scenarios` in the evaluator. Expected failures are not task successes.
- Native viewer: omit `--headless` and `MUJOCO_GL=egl`. Check camera rendering separately from viewer startup. Both have run successfully on this Linux machine despite EGL/Wayland warnings.
- Only ORACLE pose + IDEALIZED transport are implemented. Never silently map CONTACT/VISION requests to implemented modes or call mocap transport physically validated.
- World units are m/s/kg with Z up and +Y into the shelf. Tuple positions use X,Y,Z; actuator command arguments use x,z,y. Pose observations refer to the base underside center.
- Box geometry helper accepts full dimensions and divides by two for MuJoCo sizes.
- The controller receives snapshots/observations. True simulator state is accessed by adapters and the independent evaluator, not by controller logic.
- One process owns simulation state. UI callbacks enqueue commands only. Failures preserve reservations; reset creates a new scene and run directory, not physical recovery.
- Stock vision uses RGB render input, fixed reception ROI, cyan floor and orange patches. Quality is heuristic, counts are null, UNKNOWN must not resolve alerts or mean EMPTY.
- Runtime evidence is in unique ignored `runs/` directories; preserve previous logs. There is no application restart/resume support.
