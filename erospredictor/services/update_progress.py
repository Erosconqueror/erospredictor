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
        self.collection_request_changes = ()

    @classmethod
    def open(cls, root, request, resolve_patches, fresh=False, resume_completed=False, upgrade=None,
             collection_only_changes=()):
        root = Path(root)
        path = root / "state.json"
        state = None
        collection_changes = ()
        if path.is_file() and not fresh:
            try:
                state = json.loads(path.read_text(encoding="utf-8"))
            except (ValueError, UnicodeError) as exc:
                raise ValueError("Unreadable update checkpoint; use --fresh for a new update.") from exc
            if not isinstance(state, dict) or state.get("version") != 1:
                raise ValueError("Unsupported update checkpoint; use --fresh for a new update.")
            if state.get("request") != request:
                if state.get("status") != "complete":
                    previous_request = state.get("request")
                    if not isinstance(previous_request, dict):
                        raise ValueError("Invalid saved update parameters; the checkpoint was kept.")
                    changes = {key for key in set(previous_request) | set(request)
                               if previous_request.get(key) != request.get(key)}
                    collection_only = (bool(changes)
                        and changes <= set(collection_only_changes)
                        and state.get("status") in ("running", "initializing")
                        and isinstance(state.get("training"), dict) and not state["training"]
                        and not state.get("snapshot_sha256") and not state.get("publication"))
                    if collection_only:
                        # Training/QA settings have not been consumed yet. Keep
                        # the exact catalog, staging directory and pending cursor.
                        state["request"] = request
                        history = state.get("collection_request_updates", [])
                        history = history if isinstance(history, list) else []
                        collection_changes = tuple(sorted(changes))
                        state["collection_request_updates"] = (history + [list(collection_changes)])[-10:]
                    elif upgrade is None or not upgrade(state["request"], request):
                        details = ", ".join(sorted(changes))
                        raise ValueError(f"An unfinished update uses different parameters ({details}). Run with the original parameters, or --fresh.")
                    elif state.get("publication"):
                        # Finish a possibly partial installation with its original,
                        # verified sources before applying a new collection profile.
                        state["request"] = request
                    else:
                        state = {"version": 1, "run_id": uuid.uuid4().hex, "request": request,
                                 "patches": state.get("patches"), "status": "running",
                                 "collection": state.get("collection", {}), "training": {},
                                 "adopt_existing": True, "upgraded_from": state["run_id"]}
                    cls(root, state).save()
                else:
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
        result.collection_request_changes = collection_changes
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
