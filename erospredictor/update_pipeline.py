"""One resumable, non-interactive scrape -> train -> calibrate update.

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
from services.update_progress import UpdateProgress, atomic_json, file_digest
from services.update_retry import http_status, retry_delay, wait_for_retry
from services.dataset_profile import rank_targets, balanced_snapshot, collection_steps
from services.catalog_update import CatalogData, prepare_catalog, champion_dimensions

import configs as cfg


APP_DIR = Path(__file__).resolve().parent
VERSIONS_URL = "https://ddragon.leagueoflegends.com/api/versions.json"


class PermanentUpdateError(RuntimeError):
    """A configuration/data problem that waiting cannot repair."""


class UpdateLockError(PermanentUpdateError):
    pass


def compatible_profile_upgrade(previous, current):
    """Collection targets/catalog may evolve while keeping already collected data."""
    def stable(request):
        request = json.loads(json.dumps(request))
        for key in ("matches_per_tier", "players_per_tier", "pages_per_division", "tier_target",
                    "targets", "no_catalog", "champion_id.json"):
            request.pop(key, None)
        request.get("training_config", {}).pop("champion_count", None)
        return request
    return stable(previous) == stable(current)


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
            raise UpdateLockError("Another model update is running, or the update lock cannot be acquired.") from exc
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
    cli.add_argument("--matches-per-tier", type=positive_int, help="Optional flat NEW-match quota instead of the default population profile")
    cli.add_argument("--tier-target", action="append", default=[], help="Override a population target, e.g. GOLD=22000")
    cli.add_argument("--players-per-tier", type=positive_int, help="Optional player cap; default searches until the tier target")
    cli.add_argument("--pages-per-division", type=positive_int, help="Optional page cap; default follows all available pages")
    cli.add_argument("--no-catalog", action="store_true", help="Keep the installed champion catalog (for restricted/offline model updates)")
    cli.add_argument("--matches-per-player", type=positive_int, default=50)
    cli.add_argument("--min-matches", type=positive_int, default=200, help="Minimum usable records required for EVERY requested tier")
    cli.add_argument("--epochs", type=positive_int, help="Override adaptive training epochs")
    cli.add_argument("--qa-min-pass", type=float, default=0.25, help="Minimum Golden Dataset pass fraction after retry (default: 0.25)")
    mode = cli.add_mutually_exclusive_group()
    mode.add_argument("--skip-scrape", action="store_true", help="Train from selected patches already in PostgreSQL")
    mode.add_argument("--scrape-only", action="store_true", help="Collect beyond fixed quotas in profile proportions until available players are exhausted; never train")
    cli.add_argument("--dry-run", action="store_true", help="Print plan without API collection, database access or file changes")
    cli.add_argument("--fresh", action="store_true", help="Explicitly start over instead of resuming the unfinished update")
    cli.add_argument("--resume", action="store_true", help="Resume the saved update, including completion after a process crash")
    return cli


def _collection_state():
    return {"new_matches": 0, "reused_matches": 0, "visited": [], "examined_ids": [],
            "outside_window_or_invalid": 0, "division": 0, "page": 1,
            "players": None, "player_index": 0, "active_player": None,
            "match_ids": None, "match_index": 0, "pending": None, "done": False}


def _collection_report(state, target):
    return {"new_matches": state["new_matches"],
            "players": max(state.get("players_count", 0), len(state["visited"])),
            "reused_matches": state.get("reused_matches", 0), "target": target,
            "window_matches": state.get("window_matches"), "exhausted": bool(state.get("exhausted")),
            "examined": max(state.get("examined_count", 0), len(state["examined_ids"])),
            "outside_window_or_invalid": state["outside_window_or_invalid"]}


def collect_matches(args, patches, riot, db, log=print, progress=None, *, targets=None,
                    preserve_cursor=False, context=None) -> dict:
    """Persist page/player/match position; replay a DB write safely after a crash."""
    context = context if context is not None else {}
    if "known" not in context:
        context["known"] = set(db.get_all_match_ids())
        context["mapping"] = db.get_champion_mapping()
    known, mapping = context["known"], context["mapping"]
    states = progress.state["collection"] if progress else {}
    save = progress.save if progress else lambda: None
    results = {}
    targets = targets if targets is not None else rank_targets(args, cfg.DATASET_TARGETS)
    population = args.matches_per_tier is None
    riot.region, riot.cont = args.region, args.continent
    if (population or (progress and progress.state.get("adopt_existing"))) and not context.get("counts_loaded"):
        raw = db.get_all_matches(patches=patches, tiers=args.tiers, region=args.region)
        existing, _ = usable_matches(raw, mapping, patches, args.tiers, args.region)
        counts = Counter(item["tier"] for item in existing)
        for tier in args.tiers:
            state = states.setdefault(tier, _collection_state())
            goal = targets[tier]
            state["window_matches"] = counts[tier]
            credited = counts[tier] if preserve_cursor else min(goal, counts[tier])
            state["reused_matches"] = max(0, credited - state["new_matches"])
            if state["reused_matches"]:
                log(f"[Update] {tier}: {state['reused_matches']} meglévő mérkőzés átvéve a korábbi gyűjtésből.")
        if progress:
            progress.state["adopt_existing"] = False
        context["counts_loaded"] = True
        save()
    for tier in args.tiers:
        state = states.setdefault(tier, _collection_state())
        if population and state["done"] and state.get("window_matches", 0) < targets[tier]:
            if preserve_cursor:
                if not state.get("exhausted"):
                    state["done"] = False
            else:
                state.update(done=False, exhausted=False, division=0, page=1, players=None, player_index=0,
                             active_player=None, match_ids=None, match_index=0)
        divisions = ["I"] if tier in {"MASTER", "GRANDMASTER", "CHALLENGER"} else cfg.TARGET_DIVISIONS
        visited, examined = set(state["visited"]), set(state["examined_ids"])
        if state["done"]:
            log(f"{tier}: already completed; resuming the next tier.")
        while not state["done"]:
            # Intent is saved before the SQL transaction. ON CONFLICT makes replay
            # idempotent even if the process died after COMMIT but before the JSON.
            if state["pending"] is not None:
                pending = state["pending"]
                inserted = db.save_match(pending["id"], args.region, pending["data"])
                known.add(pending["id"])
                if population and inserted:
                    state["window_matches"] += 1
                state["new_matches"] += 1
                if population:
                    credited = state["window_matches"] if preserve_cursor else min(targets[tier], state["window_matches"])
                    state["reused_matches"] = max(0, credited - state["new_matches"])
                state["match_index"] += 1
                state["pending"] = None
                save()
            count = state.get("window_matches", 0) if population else state["new_matches"] + state.get("reused_matches", 0)
            if count >= targets[tier]:
                state["done"] = True
                save()
                break
            if state["division"] >= len(divisions):
                state.update(done=True, exhausted=True)
                save()
                break
            if state["players"] is None:
                players = riot.get_league_exp_players(tier, divisions[state["division"]], page=state["page"])
                if not players:
                    state["division"] += 1
                    state["page"] = 1
                    save()
                    continue
                state["players"] = [player.get("puuid") for player in players if player.get("puuid")]
                state["player_index"] = 0
                save()
            if state["player_index"] >= len(state["players"]):
                state["players"] = None
                state["page"] += 1
                if args.pages_per_division is not None and state["page"] > args.pages_per_division:
                    state["division"] += 1
                    state["page"] = 1
                save()
                continue
            puuid = state["players"][state["player_index"]]
            if state["active_player"] is None:
                if puuid in visited:
                    state["player_index"] += 1
                    save()
                    continue
                if args.players_per_tier is not None and len(visited) >= args.players_per_tier:
                    state.update(done=True, exhausted=True)
                    save()
                    break
                state["active_player"] = puuid
                visited.add(puuid)
                state["visited"] = sorted(visited)
                save()
            if state["match_ids"] is None:
                state["match_ids"] = riot.get_match_ids(puuid, limit=args.matches_per_player)
                state["match_index"] = 0
                save()
            if state["match_index"] >= len(state["match_ids"]):
                state["player_index"] += 1
                state["active_player"], state["match_ids"] = None, None
                state["match_index"] = 0
                save()
                continue
            match_id = state["match_ids"][state["match_index"]]
            if match_id in known or match_id in examined:
                state["match_index"] += 1
                save()
                continue
            data = riot.get_match_data(match_id, tier)
            examined.add(match_id)
            state["examined_ids"] = sorted(examined)
            if not data or data.get("patch") not in patches:
                state["outside_window_or_invalid"] += 1
                state["match_index"] += 1
            else:
                if population and not usable_matches([dict(data, region=args.region)], mapping,
                                                     patches, [tier], args.region)[0]:
                    state["outside_window_or_invalid"] += 1
                    state["match_index"] += 1
                    save()
                    continue
                state["pending"] = {"id": match_id, "data": data}
            save()
        state["players_count"] = max(state.get("players_count", 0), len(visited))
        state["examined_count"] = max(state.get("examined_count", 0), len(examined))
        # Completed tiers no longer need their large cursor/ID arrays in every
        # subsequent atomic checkpoint write.
        if not preserve_cursor or state.get("exhausted"):
            state.update(visited=[], examined_ids=[], players=None, match_ids=None, active_player=None)
        save()
        results[tier] = _collection_report(state, targets[tier])
        count = state.get("window_matches", 0) if population else state["new_matches"] + state.get("reused_matches", 0)
        log(f"{tier}: {count}/{targets[tier]} usable records, {state['new_matches']} new, {state['players_count']} players visited.")
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
    """Keep sources and original backups so hard-crash recovery can roll forward."""
    paths = [Path(path) for path in relative_paths]
    for relative in paths:
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid publication path.")
        if not (staging / relative).is_file():
            raise RuntimeError(f"Update output is missing: {relative}")
    backup = staging / "backup"
    journal = staging / "publication.json"
    if journal.is_file():
        previous = {Path(path) for path in json.loads(journal.read_text(encoding="utf-8"))["previous"]}
    else:
        previous = set()
        for relative in paths:
            target = destination / relative
            if target.exists():
                old = backup / relative
                old.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, old)
                previous.add(relative)
        atomic_json(journal, {"previous": [path.as_posix() for path in previous]})

    def replace_copy(source, target):
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".update-", delete=False) as output:
                temporary = Path(output.name)
                with source.open("rb") as stream:
                    shutil.copyfileobj(stream, output)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    try:
        for relative in paths:
            target = destination / relative
            replace_copy(staging / relative, target)
    except BaseException:
        for relative in reversed(paths):
            target = destination / relative
            try:
                if relative in previous:
                    replace_copy(backup / relative, target)
                else:
                    target.unlink(missing_ok=True)
            except OSError:
                # Keep all original backups for the next automatic recovery.
                continue
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
    progress = getattr(args, "_progress", None)

    view = WindowData(db, records, staging)
    stat = StatisticalModel(view)
    stat.build_stats()
    stat.save_cache()
    Preprocessor(db=view, allowed_patches=patches).gen_meta_champs()
    names = db.get_champion_names()
    dataset = GoldenDataset(str(APP_DIR / "data/golden_dataset.json"), {name: cid for cid, name in names.items()})
    if not dataset.prediction_cases or not dataset.recommendation_cases:
        raise PermanentUpdateError("Golden Dataset prediction and recommendation cases are required for QA.")
    model_dir = staging / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reports, outputs = {}, [Path("data/stats_cache.json"), Path("data/meta_champs.json")]

    for tier in list(dict.fromkeys(args.tiers)) + ["MIXED"]:
        subset = records if tier == "MIXED" else [m for m in records if m["tier"] == tier]
        prep = Preprocessor(db=WindowData(db, subset, staging), allowed_patches=patches)
        trainer = DynamicTrainer(tier, output_dir=model_dir, dataset_size=len(subset),
                                 checkpoint_dir=staging / "epochs" if progress else None)
        if args.epochs:
            trainer.config["epochs"] = args.epochs
        models, qa = {"statistical": stat}, {}
        for kind in ("roleweighted", "roleaware", "gnn"):
            key = f"{tier}:{kind}"
            cached = progress.completed_unit(key) if progress else None
            if cached:
                log(f"{tier}/{kind}: completed checkpoint verified; skipping training.")
                model = (LeagueGNN() if kind == "gnn" else ChampionPredictor() if kind == "roleweighted"
                         else RoleAwareEmbeddingPredictor()).to(device)
                path = progress.artifact(cached["path"])
                model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
                model.eval()
                models[kind], qa[kind] = model, cached["report"]
                outputs.append(Path("models") / path.name)
                continue
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
                    raise PermanentUpdateError(f"Training failed for {tier}/{kind}: {result.get('reason')}")
                model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
                model.eval()
                score = quick_evaluate_model(kind, model, tier, dataset, device)
                qa[kind] = {"golden_pass_fraction": score, "fallback": fallback, **result}
                log(f"{tier}/{kind}: Golden Dataset {score:.1%}, fallback={fallback}")
                if score >= args.qa_min_pass:
                    break
            else:
                raise PermanentUpdateError(f"Golden Dataset QA failed twice for {tier}/{kind}; existing files were kept.")
            if progress:
                progress.finish_unit(key, path, qa[kind])
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
        calibration = model_dir / f"{tier}_meta_calibrator.pkl"
        cached_calibration = progress.completed_unit(f"{tier}:calibration") if progress else None
        if cached_calibration:
            weights = cached_calibration["report"]
        else:
            weights = calibrator.calibrate(models, tier)
            calibrator.save(str(calibration))
            if progress:
                progress.finish_unit(f"{tier}:calibration", calibration, weights)
        outputs.append(Path("models") / calibration.name)
        reports[tier] = {"matches": len(subset), "qa": qa, "ensemble_weights": weights}
        del models, prep, trainer, calibrator
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return reports, outputs


def _run_scrape_only(args, targets, log=print):
    """A separate collection journal; the installed application is never published."""
    if args.matches_per_tier is not None:
        raise ValueError("Use --tier-target for --scrape-only proportions, not --matches-per-tier.")
    steps = collection_steps(targets)
    progress = None
    if args.dry_run:
        patches = resolve_patch_window(args)
    else:
        request = {key: value for key, value in vars(args).items()
                   if not key.startswith("_") and key not in
                   {"dry_run", "fresh", "resume", "epochs", "qa_min_pass", "min_matches"}}
        if args.patches:
            request["patches"] = sorted({normalize_patch(value) for value in args.patches})
        if args.end_patch:
            request["end_patch"] = normalize_patch(args.end_patch)
        request["targets"] = targets
        request["database"] = {key: getattr(cfg, key, None) for key in ("DB_NAME", "DB_HOST", "DB_PORT")}
        progress = UpdateProgress.open(APP_DIR / ".scrape_only", request,
            lambda: resolve_patch_window(args), fresh=args.fresh and not args.resume,
            resume_completed=args.resume)
        patches = progress.state["patches"]
        if progress.state["status"] == "complete":
            log("The saved scrape-only run already completed successfully.")
            return 0
    plan = {"mode": "scrape-only", "patches": patches, "region": args.region,
            "tiers": args.tiers, "profile_proportions": targets, "records_per_round": steps,
            "fixed_match_limit": None, "stages": ["collect"]}
    log(json.dumps(plan, indent=2))
    if args.dry_run:
        return 0
    log(f"[Scrape] Folytatási pont: {progress.root / 'state.json'}. Ctrl+C: leállítás.")
    from model.data_manager import DataManager
    from model.riot import Riot
    catalog, _ = prepare_catalog(APP_DIR, progress, patches, refresh=not args.no_catalog, include_icons=False)
    db = CatalogData(DataManager(dev_mode=True), catalog)
    try:
        riot = Riot(strict_errors=True, max_rate_retries=None, retry_transient=True)
        if not riot.api_key:
            raise ValueError("Set RIOT_API_KEY before collecting matches.")
        context = {}
        while True:
            states = progress.state["collection"]
            active = [tier for tier in args.tiers if not states.get(tier, {}).get("exhausted")]
            if not active:
                break
            goals = progress.state.get("scrape_goals")
            if goals is None:
                # Catch up underrepresented ranks first, considering SQL records
                # already collected in this window. Exhausted ranks cannot block others.
                level = min(states.get(tier, {}).get("window_matches", 0) // steps[tier]
                            for tier in active) + 1
                goals = {tier: (steps[tier] * level if tier in active else states[tier]["window_matches"])
                         for tier in args.tiers}
                progress.state["scrape_goals"] = goals
                progress.save()
            report = collect_matches(args, patches, riot, db, log, progress=progress,
                                     targets=goals, preserve_cursor=True, context=context)
            progress.state["collection_report"] = report
            del progress.state["scrape_goals"]
            progress.save()
        # Also reconstruct the final report after a crash between the last SQL
        # cursor checkpoint and the end-of-round report checkpoint.
        report = {tier: _collection_report(progress.state["collection"][tier], None) for tier in args.tiers}
        progress.state["collection_report"] = report
        progress.state.pop("scrape_goals", None)
        progress.save()
        summary = {"completed_at": datetime.now(timezone.utc).isoformat(), "plan": plan,
                   "collection": report, "reason": "available_players_or_search_caps_exhausted"}
        atomic_json(progress.root / "last_collection.json", summary)
        progress.complete()
        log("[Scrape] Az elérhető játékosok feldolgozva. A meccsek elmentve; a gyűjtés befejeződött.")
        return 0
    finally:
        db.close()


def _run(args, log=print) -> int:
    if args.matches_per_player > 100:
        raise ValueError("--matches-per-player cannot exceed Riot's 100-record limit.")
    if args.patches and args.end_patch:
        raise ValueError("Use --end-patch with --last-patches, not with explicit --patches.")
    args.tiers = list(dict.fromkeys(args.tiers))
    targets = rank_targets(args, cfg.DATASET_TARGETS)
    if args.scrape_only:
        return _run_scrape_only(args, targets, log)
    if args.min_matches < 4:
        raise ValueError("--min-matches must be at least 4 for training and validation.")
    if not 0 <= args.qa_min_pass <= 1:
        raise ValueError("--qa-min-pass must be between 0 and 1.")
    if args.matches_per_tier is None and any(goal < args.min_matches for goal in targets.values()):
        raise ValueError("Every population target must be at least --min-matches.")
    progress = None
    if args.dry_run:
        patches = resolve_patch_window(args)
    else:
        request = {key: value for key, value in vars(args).items()
                   if not key.startswith("_") and key not in {"dry_run", "fresh", "resume", "scrape_only"}}
        if args.patches:
            request["patches"] = sorted({normalize_patch(patch) for patch in args.patches})
        if args.end_patch:
            request["end_patch"] = normalize_patch(args.end_patch)
        request["database"] = {key: getattr(cfg, key, None) for key in ("DB_NAME", "DB_HOST", "DB_PORT")}
        request["training_config"] = {"role_weights": cfg.ROLE_WEIGHTS, "division_weights": cfg.DIVISION_WEIGHTS}
        request["targets"] = targets
        for name in ("golden_dataset.json",):
            path = APP_DIR / "data" / name
            request[name] = file_digest(path) if path.is_file() else None
        progress = UpdateProgress.open(APP_DIR / ".model_update", request,
                    lambda: resolve_patch_window(args), fresh=args.fresh and not args.resume,
                    resume_completed=args.resume, upgrade=compatible_profile_upgrade)
        args._progress = progress
        patches = progress.state["patches"]
        if progress.state["status"] == "complete":
            log("The saved update already completed successfully.")
            return 0
        if progress.state.get("publication"):
            outputs = [Path(path) for path in progress.state["publication"]]
            checksums = progress.state.get("publication_sha256", {})
            if all(progress.artifact(path).is_file() and
                   file_digest(progress.artifact(path)) == checksums.get(path.as_posix()) for path in outputs):
                log("Resuming the final model publication from its saved checkpoint...")
                publish_outputs(progress.staging, APP_DIR, outputs)
                progress.complete()
                log("Update complete.")
                return 0
            log("A végső kimenet sérült; újraépítés a mentett adatokból és modellekből.")
            del progress.state["publication"]
            progress.save()
    plan = {"patches": patches, "region": args.region, "continent": args.continent,
            "tiers": args.tiers + ["MIXED"], "scrape": not args.skip_scrape,
            "max_new_matches_per_tier": args.matches_per_tier,
            "dataset_targets": targets,
            "champion_catalog_update": not args.no_catalog,
            "max_players_per_tier": args.players_per_tier,
            "minimum_usable_matches_per_tier": args.min_matches,
            "stages": ["collect", "filter", "statistics/off-meta", "train 3 models per tier", "QA/retry", "calibrate", "publish"]}
    log(json.dumps(plan, indent=2))
    if args.dry_run:
        log("Dry run: database, models and caches were not changed.")
        return 0

    log(f"[Update] Tartós folytatási pont: {progress.root / 'state.json'}")
    from model.data_manager import DataManager
    snapshot = progress.staging / "snapshot.json"
    snapshot_ready = (snapshot.is_file() and progress.state.get("snapshot_sha256") == file_digest(snapshot))
    if progress.state.get("snapshot_sha256") and not snapshot_ready:
        raise PermanentUpdateError("Saved training data is damaged or missing. Use --fresh; existing database matches are preserved.")
    catalog, catalog_outputs = prepare_catalog(APP_DIR, progress, patches, refresh=not args.no_catalog)
    if catalog["count"] != catalog["installed_count"] and set(args.tiers) != set(cfg.TARGET_TIERS):
        raise PermanentUpdateError("New champions require updating every tier together. Omit --tiers or use --no-catalog.")
    db = CatalogData(DataManager(dev_mode=not snapshot_ready), catalog)
    try:
        if snapshot_ready:
            records = json.loads(snapshot.read_text(encoding="utf-8"))
            collection = progress.state.get("collection_report", {})
            rejected = progress.state.get("excluded_records", {})
            available = progress.state.get("available_matches", {})
            log("[Update] A mentett tanítóadatok betöltve; az adatgyűjtést nem ismétli meg.")
        else:
            collection = {}
            if not args.skip_scrape:
                from model.riot import Riot
                riot = Riot(strict_errors=True, max_rate_retries=None, retry_transient=True)
                if not riot.api_key:
                    raise ValueError("Set RIOT_API_KEY before collecting matches.")
                collection = collect_matches(args, patches, riot, db, log, progress=progress)
            raw = db.get_all_matches(patches=patches, tiers=args.tiers, region=args.region)
            records, rejected = usable_matches(raw, db.get_champion_mapping(), patches, args.tiers, args.region)
            counts = Counter(m["tier"] for m in records)
            available = dict(counts)
            log(f"Usable matches: {dict(counts)}; excluded: {rejected}")
            if rejected.get("unknown_champion"):
                log("Matches containing champions absent from the selected Data Dragon catalog were excluded.")
            if args.matches_per_tier is None:
                short = {tier: {"available": counts[tier], "target": goal}
                         for tier, goal in targets.items() if counts[tier] < goal}
                if short:
                    message = f"Population targets not reached: {short}. Collected matches were saved."
                    if args.skip_scrape or args.players_per_tier or args.pages_per_division:
                        raise PermanentUpdateError(message + " Remove collection caps or adjust --tier-target.")
                    # The next retry rescans exhausted player lists for new matches.
                    # Already stored matches stay in SQL and are never downloaded twice.
                    for tier in short:
                        progress.state["collection"][tier] = dict(_collection_state(),
                            new_matches=progress.state["collection"][tier]["new_matches"])
                    progress.save()
                    raise RuntimeError(message + " Retrying available players after a pause.")
                records = balanced_snapshot(records, targets, patches)
                log(f"Profiled training snapshot: {dict(Counter(m['tier'] for m in records))}")
            missing = {tier: counts[tier] for tier in args.tiers if counts[tier] < args.min_matches}
            if missing:
                raise PermanentUpdateError(f"Insufficient selected-patch records (need {args.min_matches} each): {missing}. No model/cache files published; collected matches remain for the next run.")
            atomic_json(snapshot, records)
            progress.state.update(snapshot_sha256=file_digest(snapshot), collection_report=collection,
                                  excluded_records=rejected, available_matches=available)
            progress.save()
        staging = progress.staging
        with champion_dimensions(catalog["count"]):
            reports, outputs = train_outputs(args, patches, db, records, staging, log)
        outputs.extend(catalog_outputs)
        runtime = {"ALLOWED_PATCHES": patches, "CHAMPION_COUNT": catalog["count"],
                   "DATASET_TARGETS": dict(cfg.DATASET_TARGETS)}
        runtime["DATASET_TARGETS"].update(targets if args.matches_per_tier is None else {})
        if catalog["version"]:
            runtime["CHAMPION_CATALOG_VERSION"] = catalog["version"]
        elif getattr(cfg, "CHAMPION_CATALOG_VERSION", None):
            runtime["CHAMPION_CATALOG_VERSION"] = cfg.CHAMPION_CATALOG_VERSION
        atomic_json(staging / "data/runtime_config.json", runtime)
        outputs.append(Path("data/runtime_config.json"))
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
                    "available_matches": available, "champion_count": catalog["count"],
                    "champion_catalog_version": catalog["version"],
                    "training_patch_counts": dict(Counter(m["patch"] for m in records)),
                    "results": reports}
        (staging / "data/update_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        outputs.append(Path("data/update_manifest.json"))
        progress.state["publication"] = [path.as_posix() for path in outputs]
        progress.state["publication_sha256"] = {path.as_posix(): file_digest(staging / path) for path in outputs}
        progress.save()
        publish_outputs(staging, APP_DIR, outputs)
        progress.complete()
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
        attempt = 0
        while True:
            try:
                return run(args)
            except (PermanentUpdateError, ValueError, ImportError) as exc:
                print(f"Update requires attention: {_safe_error(exc)}. The checkpoint was kept.", file=sys.stderr)
                return 2
            except Exception as exc:
                if http_status(exc) in {400, 401, 403, 404}:
                    print(f"Update requires attention: {_safe_error(exc)}. The checkpoint was kept.", file=sys.stderr)
                    return 2
                delay = retry_delay(attempt)
                print(f"[Update] {_safe_error(exc)}. Automatikus folytatás {delay:.0f} másodperc múlva; Ctrl+C: leállítás.", flush=True)
                args.resume = True
                wait_for_retry(delay)
                attempt += 1
    except KeyboardInterrupt:
        print("Update stopped by the user. Checkpoint saved; the next launch resumes it.", file=sys.stderr)
        return 130
    finally:
        os.chdir(previous)


def _safe_error(exc):
    message = str(exc)
    for secret in (os.environ.get("RIOT_API_KEY"), getattr(cfg, "RIOT_API_KEY", None), getattr(cfg, "DB_PASS", None)):
        if secret:
            message = message.replace(str(secret), "[redacted]")
    return message


if __name__ == "__main__":
    raise SystemExit(main())
