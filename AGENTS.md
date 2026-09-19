# Project guidance

- Run from the repository root. Use the Python 3.12 environment at `.venv`; the system `python3` is 3.14 on the development machine.
- Install exact dependencies with `uv pip install --python .venv/bin/python -r requirements-lock.txt`. Do not update dependencies during the demo.
- Verification: `MUJOCO_GL=egl .venv/bin/python -m pytest -q`.
- Integration smoke test: `MUJOCO_GL=egl .venv/bin/python -m warehouse.demo --seed 7 --perception oracle --transport idealized --headless --auto`.
- Benchmark: `MUJOCO_GL=egl .venv/bin/python -m warehouse.evaluate --seeds 0:20 --perception oracle --transport idealized`.
- Exception scenarios: `lateral missing empty unknown_stock blocked_return pick_timeout`; pass them to `--scenarios` in the evaluator. Expected failures are not task successes.
- Native viewer: omit `--headless` and `MUJOCO_GL=egl`. Check camera rendering separately from viewer startup. Both have run successfully on this Linux machine despite EGL/Wayland warnings.
- `warehouse/` is the legacy ORACLE + IDEALIZED prototype. `orderpick/` is the current Panda contact-manipulation cell; its vision mode uses RGB-D product markers but still uses oracle bin/tray poses and slip guards. Never claim complete sensory autonomy or silently substitute oracle product observations.
- Current-cell smoke: `MUJOCO_GL=egl .venv/bin/python -m orderpick.demo --seed 7 --perception vision --headless`. Current benchmark: `MUJOCO_GL=egl .venv/bin/python -m orderpick.evaluate --seeds 7:9 --modes oracle,vision --scenarios nominal`. Orderpick ranges are inclusive; exit 1 means at least one non-exact delivery.
- Current-cell success requires the independent `assessment` in `orderpick.verification`: exact SKU counts, table support, released gripper and stability over 0.5 seconds. Controller `fulfilled`/`delivered` flags alone are not sufficient.
- World units are m/s/kg with Z up and +Y into the shelf. Tuple positions use X,Y,Z; actuator command arguments use x,z,y. Pose observations refer to the base underside center.
- Box geometry helper accepts full dimensions and divides by two for MuJoCo sizes.
- Prefer snapshot/observation interfaces. Existing oracle logistics in `orderpick` are technical debt, explicitly labeled in run metadata; new product perception must not read object truth.
- One process owns simulation state. UI callbacks enqueue commands only. Failures preserve reservations; reset creates a new scene and run directory, not physical recovery.
- Stock vision uses RGB render input, fixed reception ROI, cyan floor and orange patches. Quality is heuristic, counts are null, UNKNOWN must not resolve alerts or mean EMPTY.
- Runtime evidence is in unique ignored `runs/` directories; preserve previous logs. There is no application restart/resume support.
