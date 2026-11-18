"""Four nonoverlapping collection profiles, including shared champion-trait metrics."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time

from services.api_keys import PROFILES, get_api_key
from regional_collector import ROUTES, PROFILE_REGIONS, stop_workers
import update_pipeline as pipeline

DEFAULT_ASSIGNMENTS = PROFILE_REGIONS


def validate_plan(plan):
    seen = set()
    if not isinstance(plan, dict) or not plan:
        raise ValueError("Collection plan must map profiles to platforms")
    for profile, regions in plan.items():
        if profile not in PROFILES or not isinstance(regions, list) or not regions:
            raise ValueError("Invalid key profile or empty region list")
        routes = set()
        for region in regions:
            if region not in ROUTES or region in seen:
                raise ValueError("Each platform must belong to exactly one key profile")
            if ROUTES[region] in routes:
                raise ValueError("Use one platform per routing budget per profile; no forced Europe split")
            seen.add(region)
            routes.add(ROUTES[region])
    return plan


def run(plan, hours, extra, dry_run=False):
    validate_plan(plan)
    if not math.isfinite(hours) or hours <= 0 or hours > 24:
        raise ValueError("Run duration must be between 0 and 24 hours")
    parsed = pipeline.parser().parse_args(extra)
    if parsed.skip_scrape or parsed.all_matches or parsed.all_regions:
        raise ValueError("The daily launcher only collects")
    if dry_run:
        patches = pipeline.resolve_patch_window(parsed)
        print(json.dumps({"hours": hours, "patches": patches, "assignments": plan,
            "match_history": "100 IDs per page, at most 10 pages per player unless overridden",
            "regions_overlap": False}, indent=2))
        return 0
    keys = [get_api_key(profile) for profile in plan]
    if not all(keys):
        raise ValueError("Fill all requested key profiles in secrets.local.json before launching the day plan")
    if len(set(keys)) != len(keys):
        raise ValueError("Profiles must use distinct keys")
    from contextlib import ExitStack
    with ExitStack() as locks:
        for profile in plan:
            filename = ".model_update.lock" if profile == "default" else f".model_update.{profile}.lock"
            locks.enter_context(pipeline.UpdateLock(pipeline.APP_DIR / filename))
    workers = {}
    root = pipeline.APP_DIR / ".scrape_regions/day-logs"
    root.mkdir(parents=True, exist_ok=True)
    try:
        for profile, regions in plan.items():
            output = (root / f"{profile}.log").open("a", encoding="utf-8")
            command = [sys.executable, "-u", str(Path(__file__).with_name("regional_collector.py")),
                "--key-profile", profile, "--regions", *regions, "--run-hours", str(hours),
                "--matches-per-player", "100", "--history-pages", "10", *extra]
            try:
                process = subprocess.Popen(command, cwd=pipeline.APP_DIR, stdout=output, stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0)
            except BaseException:
                output.close()
                raise
            workers[profile] = process, output
            print(f"{profile}: {', '.join(regions)}. Log: {root / (profile + '.log')}", flush=True)
        deadline = time.monotonic() + hours * 3600
        warned = set()
        while time.monotonic() < deadline and any(p.poll() is None for p, _ in workers.values()):
            for profile, (process, _) in workers.items():
                if process.poll() not in (None, 0) and profile not in warned:
                    print(f"{profile} stopped with exit {process.returncode}; other profiles continue. See its log.", flush=True)
                    warned.add(profile)
            time.sleep(1)
        return 2 if warned or any(p.poll() not in (None, 0) for p, _ in workers.values()) else 0
    finally:
        stop_workers(workers)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=float, default=24)
    parser.add_argument("--plan", type=Path, help="JSON profile -> region list mapping")
    parser.add_argument("--dry-run", action="store_true")
    args, extra = parser.parse_known_args(argv)
    try:
        plan = json.loads(args.plan.read_text(encoding="utf-8")) if args.plan else DEFAULT_ASSIGNMENTS
        return run(plan, args.hours, extra, args.dry_run)
    except KeyboardInterrupt:
        print("Day run stopped. Checkpoints retained.")
        return 130
    except (ValueError, OSError, pipeline.UpdateLockError) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
