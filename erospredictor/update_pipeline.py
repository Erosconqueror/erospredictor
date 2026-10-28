"""One finite, non-interactive scrape -> train -> calibrate update.

Run from anywhere with ``python erospredictor/update_pipeline.py --help``.
Heavy dependencies and the database are opened only after the update plan is
resolved. A failed run keeps the currently published model/cache files intact;
already collected matches remain in PostgreSQL so collection can be resumed.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from urllib.request import urlopen

import configs as cfg


APP_DIR = Path(__file__).resolve().parent
VERSIONS_URL = "https://ddragon.leagueoflegends.com/api/versions.json"


class UpdateLock:
    """OS-managed exclusive lock; a crash releases it without stale-PID guessing."""

    def __init__(self, path):
        self.path = Path(path)
        self.handle = None

    def __enter__(self):
        descriptor = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        self.handle = os.fdopen(descriptor, "r+b")
        try:
            if self.path.stat().st_size == 0:
                self.handle.write(b"0")
                self.handle.flush()
            self.handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.handle.close()
            self.handle = None
            raise RuntimeError("Another model update is running, or the update lock cannot be acquired.") from exc
        try:
            self.handle.seek(1)
            self.handle.truncate()
            self.handle.write(json.dumps({"pid": os.getpid(), "started_at": datetime.now(timezone.utc).isoformat()}).encode("utf-8"))
            self.handle.flush()
        except BaseException:
            self.handle.close()
            self.handle = None
            raise
        return self

    def __exit__(self, *_):
        if self.handle is not None:
            try:
                self.handle.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            finally:
                self.handle.close()
                self.handle = None


def normalize_patch(value: str) -> str:
    """Use Match-v5/Data Dragon versions; public 26.x names alias internal 16.x."""
    match = re.fullmatch(r"(\d+)\.(\d+)(?:\.\d+)*", value.strip())
    if not match:
        raise ValueError(f"Invalid patch: {value!r}; expected e.g. 16.7 or 26.7.")
    major, minor = map(int, match.groups())
    if major >= 25:
        major -= 10
    if major < 1 or minor < 1:
        raise ValueError(f"Invalid patch: {value!r}.")
    return f"{major}.{minor}"


def fetch_versions() -> list[str]:
    with urlopen(VERSIONS_URL, timeout=15) as response:
        versions = json.load(response)
    if not isinstance(versions, list) or not versions:
        raise ValueError("Data Dragon returned an empty or invalid version list.")
    return versions


def resolve_patch_window(args, versions=None) -> list[str]:
    """Resolve actual releases (including year boundaries and skipped patches)."""
    if args.patches:
        return sorted({normalize_patch(p) for p in args.patches},
                      key=lambda p: tuple(map(int, p.split("."))))
    releases = []
    for version in fetch_versions() if versions is None else versions:
        try:
            patch = normalize_patch(version)
        except (ValueError, AttributeError):
            continue
        if patch not in releases:
            releases.append(patch)
    if args.end_patch:
        anchor = normalize_patch(args.end_patch)
        if anchor not in releases:
            raise ValueError(f"Patch {anchor} is absent from Data Dragon's releases.")
        releases = releases[releases.index(anchor):]
    if len(releases) < args.last_patches:
        raise ValueError("Not enough released patches to resolve the requested window.")
    return list(reversed(releases[:args.last_patches]))


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    window = cli.add_mutually_exclusive_group()
    window.add_argument("--patches", nargs="+", help="Exact patches, e.g. 16.5 16.6 16.7 (26.x public names also accepted)")
    window.add_argument("--last-patches", type=positive_int, default=5, help="Number of most recent released patches (default: 5)")
    cli.add_argument("--end-patch", help="End automatic release window here instead of the latest release")
    cli.add_argument("--tiers", nargs="+", type=str.upper, choices=cfg.TARGET_TIERS, default=list(cfg.TARGET_TIERS))
    cli.add_argument("--region", default=cfg.REGION, help="Platform route, e.g. euw1, eun1, kr")
    cli.add_argument("--continent", choices=("americas", "asia", "europe", "sea"), default=cfg.CONTINENT)
    cli.add_argument("--matches-per-tier", type=positive_int, default=2000, help="Maximum NEW matching records collected per tier (default: 2000)")
    cli.add_argument("--players-per-tier", type=positive_int, default=200, help="Maximum unique players visited per tier (default: 200)")
    cli.add_argument("--pages-per-division", type=positive_int, default=2)
    cli.add_argument("--matches-per-player", type=positive_int, default=50)
    cli.add_argument("--min-matches", type=positive_int, default=200, help="Minimum usable records required for EVERY requested tier")
    cli.add_argument("--epochs", type=positive_int, help="Override adaptive training epochs")
    cli.add_argument("--qa-min-pass", type=float, default=0.25, help="Minimum Golden Dataset pass fraction after retry (default: 0.25)")
    cli.add_argument("--skip-scrape", action="store_true", help="Train from selected patches already in PostgreSQL")
    cli.add_argument("--dry-run", action="store_true", help="Print plan without API collection, database access or file changes")
    return cli


def collect_matches(args, patches, riot, db, log=print) -> dict:
    """A bounded pass, unlike the legacy collector's continuous while loop."""
    known = set(db.get_all_match_ids())
    results = {}
    riot.region, riot.cont = args.region, args.continent
    for tier in args.tiers:
        saved, visited, examined, rejected = 0, set(), set(), 0
        divisions = ["I"] if tier in {"MASTER", "GRANDMASTER", "CHALLENGER"} else cfg.TARGET_DIVISIONS
        stop = False
        for division in divisions:
            if stop:
                break
            for page in range(1, args.pages_per_division + 1):
                players = riot.get_league_exp_players(tier, division, page=page)
                if not players:
                    break
                for player in players:
                    puuid = player.get("puuid")
                    if not puuid or puuid in visited:
                        continue
                    if len(visited) >= args.players_per_tier or saved >= args.matches_per_tier:
                        stop = True
                        break
                    visited.add(puuid)
                    for match_id in riot.get_match_ids(puuid, limit=args.matches_per_player):
                        if saved >= args.matches_per_tier:
                            stop = True
                            break
                        if match_id in known or match_id in examined:
                            continue
                        examined.add(match_id)
                        data = riot.get_match_data(match_id, tier)
                        if not data or data.get("patch") not in patches:
                            rejected += 1
                            continue
                        if db.save_match(match_id, args.region, data):
                            saved += 1
                        known.add(match_id)
                if stop:
                    break
        results[tier] = {"new_matches": saved, "players": len(visited),
                         "examined": len(examined), "outside_window_or_invalid": rejected}
        log(f"{tier}: {saved} new matching records, {len(visited)} players visited.")
    return results


def usable_matches(records, mapping, patches, tiers, region):
    """Exclude unknown champions and malformed records consistently for all outputs."""
    usable, rejected = [], Counter()
    for original in records:
        if (original.get("patch") not in patches or original.get("tier") not in tiers
                or original.get("region") != region):
            rejected["outside_window"] += 1
            continue
        record = dict(original)
        for name in ("blue_team", "red_team"):
            team = record.get(name, [])
            if isinstance(team, str):
                team = team.strip("{}").split(",")
            record[name] = list(team) if isinstance(team, (list, tuple)) else []
        teams = record["blue_team"] + record["red_team"]
        if len(record["blue_team"]) != 5 or len(record["red_team"]) != 5 or len(set(map(str, teams))) != 10:
            rejected["invalid_teams"] += 1
        elif any(str(champion) not in mapping for champion in teams):
            rejected["unknown_champion"] += 1
        elif not isinstance(record.get("blue_win"), bool):
            rejected["invalid_outcome"] += 1
        else:
            usable.append(record)
    return usable, dict(rejected)


class WindowData:
    """One immutable training snapshot, with derived JSON directed into staging."""

    def __init__(self, db, records, staging):
        self.db, self.records, self.staging = db, records, Path(staging)

    def get_all_matches(self):
        return self.records

    def get_champion_mapping(self):
        return self.db.get_champion_mapping()

    def _save(self, data, name):
        path = self.staging / "data" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def save_stats_cache(self, data, path=None):
        self._save(data, "stats_cache.json")

    def save_meta_champs(self, data, path=None):
        self._save(data, "meta_champs.json")


def publish_outputs(staging: Path, destination: Path, relative_paths):
    """Publish only a complete run, rolling back any file-replacement failure."""
    paths = [Path(path) for path in relative_paths]
    for relative in paths:
        if not (staging / relative).is_file():
            raise RuntimeError(f"Update output is missing: {relative}")
    backup = staging / "backup"
    previous, replaced = set(), []
    for relative in paths:
        target = destination / relative
        if target.exists():
            old = backup / relative
            old.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, old)
            previous.add(relative)
    try:
        for relative in paths:
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging / relative, target)
            replaced.append(relative)
    except BaseException:
        for relative in reversed(replaced):
            target = destination / relative
            if relative in previous:
                os.replace(backup / relative, target)
            else:
                target.unlink(missing_ok=True)
        raise


def train_outputs(args, patches, db, records, staging, log=print):
    # Import only here: help and explicit dry-runs work without the ML packages.
    import torch
    from model.preprocessor import Preprocessor
    from model.train_model import DynamicTrainer
    from model.statistical import StatisticalModel
    from model.meta_calibrator import MetaLearningCalibrator
    from model.golden_dataset import GoldenDataset
    from model.gnn_predictor import LeagueGNN
    from model.predictor import ChampionPredictor, RoleAwareEmbeddingPredictor
    from trainer_script import quick_evaluate_model

    view = WindowData(db, records, staging)
    stat = StatisticalModel(view)
    stat.build_stats()
    stat.save_cache()
    Preprocessor(db=view, allowed_patches=patches).gen_meta_champs()
    names = db.get_champion_names()
    dataset = GoldenDataset(str(APP_DIR / "data/golden_dataset.json"), {name: cid for cid, name in names.items()})
    if not dataset.prediction_cases or not dataset.recommendation_cases:
        raise RuntimeError("Golden Dataset prediction and recommendation cases are required for QA.")
    model_dir = staging / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reports, outputs = {}, [Path("data/stats_cache.json"), Path("data/meta_champs.json")]

    for tier in list(dict.fromkeys(args.tiers)) + ["MIXED"]:
        subset = records if tier == "MIXED" else [m for m in records if m["tier"] == tier]
        prep = Preprocessor(db=WindowData(db, subset, staging), allowed_patches=patches)
        trainer = DynamicTrainer(tier, output_dir=model_dir, dataset_size=len(subset))
        if args.epochs:
            trainer.config["epochs"] = args.epochs
        models, qa = {"statistical": stat}, {}
        for kind in ("roleweighted", "roleaware", "gnn"):
            log(f"Training {tier} / {kind} on {len(subset)} selected matches...")
            if kind == "gnn":
                graphs, _ = prep.process_matches_gnn()
                fit = lambda fallback=False: trainer.train_gnn_model(graphs, f"{tier}_{kind}", fallback=fallback)
                model = LeagueGNN().to(device)
            else:
                process = prep.process_matches if kind == "roleweighted" else prep.process_matches_ra
                x, y, _, weights = process()
                size = cfg.CHAMPION_COUNT * (2 if kind == "roleweighted" else 10)
                model_type = "standard" if kind == "roleweighted" else "roleaware"
                fit = lambda fallback=False: trainer.train_single_model(x, y, f"{tier}_{kind}", size, model_type, weights, fallback=fallback)
                model = (ChampionPredictor(input_size=size) if kind == "roleweighted" else RoleAwareEmbeddingPredictor()).to(device)
            path = model_dir / f"{tier}_{kind}.pth"
            for fallback in (False, True):
                result = fit(fallback=fallback)
                if result.get("status") != "success":
                    raise RuntimeError(f"Training failed for {tier}/{kind}: {result.get('reason')}")
                model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
                model.eval()
                score = quick_evaluate_model(kind, model, tier, dataset, device)
                qa[kind] = {"golden_pass_fraction": score, "fallback": fallback, **result}
                log(f"{tier}/{kind}: Golden Dataset {score:.1%}, fallback={fallback}")
                if score >= args.qa_min_pass:
                    break
            else:
                raise RuntimeError(f"Golden Dataset QA failed twice for {tier}/{kind}; existing files were kept.")
            models[kind] = model
            outputs.append(Path("models") / path.name)
            prep.clear_cache()
            del fit
            if kind == "gnn":
                del graphs
            else:
                del x, y, weights
            gc.collect()
        calibrator = MetaLearningCalibrator(dataset=dataset)
        weights = calibrator.calibrate(models, tier)
        calibration = model_dir / f"{tier}_meta_calibrator.pkl"
        calibrator.save(str(calibration))
        outputs.append(Path("models") / calibration.name)
        reports[tier] = {"matches": len(subset), "qa": qa, "ensemble_weights": weights}
        del models, prep, trainer, calibrator
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return reports, outputs


def _run(args, log=print) -> int:
    if args.matches_per_player > 100:
        raise ValueError("--matches-per-player cannot exceed Riot's 100-record limit.")
    if args.min_matches < 4:
        raise ValueError("--min-matches must be at least 4 for training and validation.")
    if not 0 <= args.qa_min_pass <= 1:
        raise ValueError("--qa-min-pass must be between 0 and 1.")
    if args.patches and args.end_patch:
        raise ValueError("Use --end-patch with --last-patches, not with explicit --patches.")
    args.tiers = list(dict.fromkeys(args.tiers))
    patches = resolve_patch_window(args)
    plan = {"patches": patches, "region": args.region, "continent": args.continent,
            "tiers": args.tiers + ["MIXED"], "scrape": not args.skip_scrape,
            "max_new_matches_per_tier": args.matches_per_tier,
            "max_players_per_tier": args.players_per_tier,
            "minimum_usable_matches_per_tier": args.min_matches,
            "stages": ["collect", "filter", "statistics/off-meta", "train 3 models per tier", "QA/retry", "calibrate", "publish"]}
    log(json.dumps(plan, indent=2))
    if args.dry_run:
        log("Dry run: database, models and caches were not changed.")
        return 0

    from model.data_manager import DataManager
    db = DataManager(dev_mode=True)
    try:
        collection = {}
        if not args.skip_scrape:
            from model.riot import Riot
            riot = Riot(strict_errors=True)
            if not riot.api_key:
                raise ValueError("Set RIOT_API_KEY before collecting matches.")
            collection = collect_matches(args, patches, riot, db, log)
        raw = db.get_all_matches(patches=patches, tiers=args.tiers, region=args.region)
        records, rejected = usable_matches(raw, db.get_champion_mapping(), patches, args.tiers, args.region)
        counts = Counter(m["tier"] for m in records)
        log(f"Usable matches: {dict(counts)}; excluded: {rejected}")
        if rejected.get("unknown_champion"):
            log("WARNING: Matches containing champions absent from the installed catalog were excluded. This run does not add new champions to existing model architectures.")
        missing = {tier: counts[tier] for tier in args.tiers if counts[tier] < args.min_matches}
        if missing:
            raise RuntimeError(f"Insufficient selected-patch records (need {args.min_matches} each): {missing}. No model/cache files published; collected matches remain for the next run.")
        with tempfile.TemporaryDirectory(prefix=".update-", dir=APP_DIR) as directory:
            staging = Path(directory)
            reports, outputs = train_outputs(args, patches, db, records, staging, log)
            # A deliberately restricted --tiers run must not remove the lookup
            # data for other, unchanged tiers in the installed application.
            for name in ("stats_cache.json", "meta_champs.json"):
                old_path, new_path = APP_DIR / "data" / name, staging / "data" / name
                if old_path.is_file() and new_path.is_file():
                    previous_cache = json.loads(old_path.read_text(encoding="utf-8"))
                    new_cache = json.loads(new_path.read_text(encoding="utf-8"))
                    untouched = {key: value for key, value in previous_cache.items()
                                 if key not in set(args.tiers) | {"MIXED", "__scope__"}}
                    new_path.write_text(json.dumps({**untouched, **new_cache}, indent=2), encoding="utf-8")
            stats_path = staging / "data/stats_cache.json"
            if stats_path.is_file():
                scoped_stats = json.loads(stats_path.read_text(encoding="utf-8"))
                scoped_stats["__scope__"] = {"mixed_tiers": args.tiers, "patches": patches}
                stats_path.write_text(json.dumps(scoped_stats, indent=2), encoding="utf-8")
            manifest = {"completed_at": datetime.now(timezone.utc).isoformat(),
                        "plan": plan, "collection": collection, "excluded_records": rejected,
                        "results": reports}
            (staging / "data/update_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            outputs.append(Path("data/update_manifest.json"))
            publish_outputs(staging, APP_DIR, outputs)
        log("Update complete: all requested tiers, MIXED, caches and calibration were published.")
        return 0
    finally:
        db.close()


def run(args, log=print) -> int:
    if args.dry_run:
        return _run(args, log)
    with UpdateLock(APP_DIR / ".model_update.lock"):
        return _run(args, log)


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    previous = Path.cwd()
    try:
        os.chdir(APP_DIR)
        return run(args)
    except KeyboardInterrupt:
        print("Update interrupted; current published files were kept.", file=sys.stderr)
        return 130
    except Exception as exc:
        # API exceptions may include request headers; never print credentials.
        message = str(exc)
        for secret in (os.environ.get("RIOT_API_KEY"), getattr(cfg, "RIOT_API_KEY", None), getattr(cfg, "DB_PASS", None)):
            if secret:
                message = message.replace(str(secret), "[redacted]")
        print(f"Update failed: {message}", file=sys.stderr)
        return 1
    finally:
        os.chdir(previous)


if __name__ == "__main__":
    raise SystemExit(main())
