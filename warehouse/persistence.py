from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from uuid import uuid4

from warehouse.contracts import Event
from warehouse.scene import ROOT


def code_version():
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    digest = hashlib.sha256()
    files = sorted((ROOT / "warehouse").glob("*.py"))
    files += sorted((ROOT / "orderpick").glob("*.py"))
    files += sorted((ROOT / "config").rglob("*.json"))
    files += [ROOT / "requirements-lock.txt"]
    for path in files:
        if path.exists():
            digest.update(str(path.relative_to(ROOT)).encode())
            digest.update(path.read_bytes())
    return {"git_commit": result.stdout.strip() if result.returncode == 0 else None, "source_sha256": digest.hexdigest()}


class RunLog:
    def __init__(self, root="runs", metadata=None):
        self.path = Path(root) / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{uuid4().hex[:10]}"
        self.path.mkdir(parents=True, exist_ok=False)
        self.started = time.monotonic()
        self.metadata = {**code_version(), **(metadata or {}), "started_utc": datetime.now(timezone.utc).isoformat()}
        self.write_json("metadata.json", self.metadata)

    def write_json(self, name, value):
        destination = self.path / name
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        with temporary.open("w") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(destination)

    def record(self, request_id, event_type, sim_time_s, payload, inventory):
        event = Event(request_id, event_type, sim_time_s, time.monotonic() - self.started, payload)
        record = {**asdict(event), "observed_utc": datetime.now(timezone.utc).isoformat(), "inventory": inventory.snapshot()}
        with (self.path / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.write_json("inventory.json", inventory.snapshot())
