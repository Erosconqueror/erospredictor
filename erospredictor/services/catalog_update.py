"""Stage an append-only Data Dragon catalog together with compatible models."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from urllib.request import urlopen
import zlib

from services.update_progress import atomic_json, file_digest


CDN = "https://ddragon.leagueoflegends.com"


def fetch_json(url):
    with urlopen(url, timeout=20) as response:
        try:
            return json.load(response)
        except (ValueError, UnicodeError) as exc:
            raise ConnectionError("Incomplete Data Dragon response; retrying") from exc


def version_for_patch(patch, versions):
    candidates = [value for value in versions if isinstance(value, str)
                  and re.fullmatch(r"\d+\.\d+\.\d+", value) and value.rsplit(".", 1)[0] == patch]
    if not candidates:
        raise ValueError(f"No Data Dragon champion catalog exists for patch {patch}")
    return max(candidates, key=lambda value: tuple(map(int, value.split("."))))


def merge_catalog(mapping, names, document):
    """Existing indices, especially Aatrox=0, never change; new IDs append."""
    mapping = {str(key): int(value) for key, value in mapping.items()}
    names = {str(key): value for key, value in names.items()}
    indices = set(mapping.values())
    if (not indices or indices != set(range(max(indices) + 1)) or len(indices) != len(mapping)
            or set(names) != {str(index) for index in indices}):
        raise ValueError("The installed champion IDs/names are inconsistent")
    data = document.get("data") if isinstance(document, dict) else None
    if not isinstance(data, dict) or not data:
        raise ValueError("Invalid Data Dragon champion list")
    entries = []
    for item in data.values():
        try:
            riot_id, name, filename = str(int(item["key"])), item["name"], item["image"]["full"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Invalid champion catalog entry") from exc
        if int(riot_id) <= 0 or not isinstance(name, str) or not name.strip():
            raise ValueError("Invalid champion ID or name")
        if not isinstance(filename, str) or not re.fullmatch(r"[A-Za-z0-9_-]+\.png", filename):
            raise ValueError("Invalid champion icon filename")
        entries.append((riot_id, name, filename))
    if len({item[0] for item in entries}) != len(entries):
        raise ValueError("Duplicate Riot champion ID in catalog")
    images = []
    for riot_id, name, filename in sorted(entries, key=lambda item: int(item[0])):
        if riot_id not in mapping:
            mapping[riot_id] = len(mapping)
        index = mapping[riot_id]
        names[str(index)] = name
        images.append({"index": index, "filename": filename})
    return {"mapping": mapping, "names": names, "count": len(mapping), "images": images}


def _write_bytes(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".catalog-", delete=False) as output:
            temporary = Path(output.name)
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def complete_png(payload):
    """Reject truncated downloads even when the PNG header arrived intact."""
    if payload[:8] != b"\x89PNG\r\n\x1a\n":
        return False
    offset, has_header, has_data = 8, False, False
    while offset + 12 <= len(payload):
        size = int.from_bytes(payload[offset:offset + 4], "big")
        end = offset + 12 + size
        if end > len(payload):
            return False
        kind, data = payload[offset + 4:offset + 8], payload[offset + 8:end - 4]
        if zlib.crc32(kind + data) != int.from_bytes(payload[end - 4:end], "big"):
            return False
        if not has_header:
            if kind != b"IHDR" or size != 13 or not all(int.from_bytes(data[i:i + 4], "big") for i in (0, 4)):
                return False
            has_header = True
        if kind == b"IDAT" and data:
            has_data = True
        if kind == b"IEND":
            return size == 0 and has_data and end == len(payload)
        offset = end
    return False


def prepare_catalog(app_dir, progress, patches, refresh=True):
    app_dir = Path(app_dir)
    snapshot = progress.staging / "catalog.json"
    expected = progress.state.get("catalog_sha256")
    if expected and (not snapshot.is_file() or expected != file_digest(snapshot)):
        raise ValueError("Saved champion catalog is damaged or missing. Use --fresh; SQL matches are preserved.")
    if expected:
        catalog = json.loads(snapshot.read_text(encoding="utf-8"))
    else:
        mapping = json.loads((app_dir / "data/champion_id.json").read_text(encoding="utf-8"))
        names = json.loads((app_dir / "data/champion_names.json").read_text(encoding="utf-8"))
        if refresh:
            versions = fetch_json(CDN + "/api/versions.json")
            version = version_for_patch(patches[-1], versions)
            document = fetch_json(f"{CDN}/cdn/{version}/data/en_US/champion.json")
            catalog = dict(merge_catalog(mapping, names, document), version=version, refresh=True)
        else:
            catalog = {"mapping": mapping, "names": names,
                       "count": max(int(value) for value in mapping.values()) + 1,
                       "images": [], "version": None, "refresh": False}
        catalog["installed_count"] = len(mapping)
        atomic_json(snapshot, catalog)
        progress.state["catalog_sha256"] = file_digest(snapshot)
        progress.save()
    outputs = []
    if catalog["refresh"]:
        for name, value in (("champion_id.json", catalog["mapping"]), ("champion_names.json", catalog["names"])):
            relative = Path("data") / name
            atomic_json(progress.staging / relative, value)
            outputs.append(relative)
        for image in catalog["images"]:
            relative = Path("assets/icons") / f"{image['index']}.png"
            if (app_dir / relative).is_file():
                continue
            target = progress.staging / relative
            if not target.is_file() or not complete_png(target.read_bytes()):
                with urlopen(f"{CDN}/cdn/{catalog['version']}/img/champion/{image['filename']}", timeout=20) as response:
                    payload = response.read()
                if not complete_png(payload):
                    raise ConnectionError("Incomplete champion icon; retrying")
                _write_bytes(target, payload)
            outputs.append(relative)
    return catalog, outputs


class CatalogData:
    def __init__(self, database, catalog):
        self.database, self.catalog = database, catalog

    def get_champion_mapping(self):
        return self.catalog["mapping"]

    def get_champion_names(self):
        return {int(key): name for key, name in self.catalog["names"].items()}

    def __getattr__(self, name):
        return getattr(self.database, name)


@contextmanager
def champion_dimensions(count):
    """Also cover modules already imported before an automatic in-process retry."""
    import configs
    modules = ("configs", "trainer_script", "model.predictor", "model.gnn_predictor",
               "model.preprocessor", "model.train_model", "model.meta_calibrator", "model.core_model")
    previous_count = configs.CHAMPION_COUNT
    previous = {name: sys.modules[name].CHAMPION_COUNT for name in modules
                if name in sys.modules and hasattr(sys.modules[name], "CHAMPION_COUNT")}
    for name in modules:
        if name in sys.modules:
            sys.modules[name].CHAMPION_COUNT = count
    try:
        yield
    finally:
        for name in modules:
            if name in sys.modules:
                sys.modules[name].CHAMPION_COUNT = previous.get(name, previous_count)
