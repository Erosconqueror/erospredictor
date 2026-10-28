"""Durable update journal and staged artifacts, scoped to one local update."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import uuid


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".checkpoint-", delete=False) as output:
            temporary = Path(output.name)
            json.dump(value, output, ensure_ascii=False)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class UpdateProgress:
    def __init__(self, root, state):
        self.root, self.state = Path(root), state
        run_id = state.get("run_id", "")
        if not isinstance(run_id, str) or len(run_id) != 32 or any(c not in "0123456789abcdef" for c in run_id):
            raise ValueError("Invalid update checkpoint; use --fresh to start a new update.")
        self.staging = self.root / ("run-" + run_id)

    @classmethod
    def open(cls, root, request, resolve_patches, fresh=False, resume_completed=False):
        root = Path(root)
        path = root / "state.json"
        state = None
        if path.is_file() and not fresh:
            try:
                state = json.loads(path.read_text(encoding="utf-8"))
            except (ValueError, UnicodeError) as exc:
                raise ValueError("Unreadable update checkpoint; use --fresh for a new update.") from exc
            if not isinstance(state, dict) or state.get("version") != 1:
                raise ValueError("Unsupported update checkpoint; use --fresh for a new update.")
            if state.get("request") != request:
                if state.get("status") != "complete":
                    raise ValueError("An unfinished update uses different parameters. Run with the original parameters, or --fresh.")
                state = None
            elif state.get("status") == "complete" and not resume_completed:
                state = None
        if state is None:
            state = {"version": 1, "run_id": uuid.uuid4().hex, "request": request,
                     "patches": None, "status": "initializing", "collection": {}, "training": {},
                     "adopt_existing": not path.is_file() and not fresh}
            result = cls(root, state)
            result.save()
        else:
            result = cls(root, state)
        if result.state.get("patches") is None:
            result.state["patches"] = resolve_patches()
            result.state["status"] = "running"
            result.save()
        if result.state["status"] != "complete":
            result.staging.mkdir(parents=True, exist_ok=True)
        return result

    def save(self):
        atomic_json(self.root / "state.json", self.state)

    def artifact(self, relative):
        relative = Path(relative)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid checkpoint artifact path.")
        return self.staging / relative

    def completed_unit(self, key):
        unit = self.state["training"].get(key)
        if unit:
            path = self.artifact(unit["path"])
            if path.is_file() and file_digest(path) == unit["sha256"]:
                return unit
        return None

    def finish_unit(self, key, path, report):
        path = Path(path)
        self.state["training"][key] = {"path": path.relative_to(self.staging).as_posix(),
                                       "sha256": file_digest(path), "report": report}
        self.save()

    def complete(self):
        self.state["status"] = "complete"
        self.save()
        # Only this journal's generated run directory may be removed.
        target, parent = self.staging.resolve(), self.root.resolve()
        if target.parent != parent or target.name != "run-" + self.state["run_id"]:
            raise ValueError("Invalid update staging directory.")
        if target.exists():
            shutil.rmtree(target)
