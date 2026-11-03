"""One supervisor, independent Match-v5 routing budgets, durable per-region cursors."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import threading
import _thread

from services.api_keys import get_api_key
import update_pipeline as pipeline

ROUTES = {"euw1": "europe", "eun1": "europe", "kr": "asia", "na1": "americas"}


def selected_routes(regions):
    regions = list(dict.fromkeys(regions))
    if not regions or any(region not in ROUTES for region in regions):
        raise ValueError("Supported regions: euw1, eun1, kr, na1")
    return regions


def route_groups(regions):
    groups = {}
    for region in selected_routes(regions):
        groups.setdefault(ROUTES[region], []).append(region)
    return list(groups.values())


def collected_count(region):
    try:
        state = json.loads((pipeline.APP_DIR / ".scrape_regions" / region / "state.json").read_text(encoding="utf-8"))
        return sum(row.get("new_matches", 0) for row in state.get("collection", {}).values())
    except (OSError, ValueError, TypeError):
        return 0


def collect_group(regions, extra):
    """One process per routing budget; favour the region with fewer saved new games."""
    if len({ROUTES[region] for region in regions}) != 1:
        raise ValueError("A worker group must share one routing budget")
    pipeline.REGIONAL_ROUND = len(regions) > 1
    completed, started = {}, set()
    while len(completed) < len(regions):
        active = [region for region in regions if region not in completed]
        region = min(active, key=collected_count)
        pipeline.REGIONAL_WORKER = region
        values = list(extra)
        if region in started:
            values = [value for value in values if value != "--fresh"]
        started.add(region)
        print(f"[{ROUTES[region]} rotation] {region}: {collected_count(region)} new saved games", flush=True)
        code = pipeline.main(worker_arguments(region, values))
        if code == 130:
            return code
        if code != pipeline.ROUND_YIELDED:
            completed[region] = code
    return max(completed.values(), default=0)


def worker_arguments(region, extra):
    # Worker scopes cannot be overridden by passthrough arguments.
    parsed = pipeline.parser().parse_args(extra)
    if parsed.skip_scrape or parsed.all_regions:
        raise ValueError("This launcher only collects matches")
    return [*extra, "--scrape-only", "--region", region, "--continent", ROUTES[region]]


def stop_workers(workers):
    for process, stream in workers.values():
        if process.poll() is None:
            try:
                process.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT)
            except OSError:
                pass
    deadline = time.monotonic() + 15
    for process, stream in workers.values():
        if process.poll() is None:
            try:
                process.wait(timeout=max(.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                process.terminate()  # SQL/journal replay remains idempotent after abrupt termination.
                process.wait()
        stream.close()
        if process.stdin is not None:
            process.stdin.close()
    workers.clear()


@contextmanager
def collector_lock(workers):
    with pipeline.UpdateLock(pipeline.APP_DIR / ".model_update.lock"):
        try:
            yield
        finally:
            stop_workers(workers)


def supervise(regions, profile, extra):
    key = get_api_key(profile)
    if not key:
        raise ValueError(f"Missing {profile} key. Fill secrets.local.json or its environment variable.")
    groups = {group[0]: group for group in route_groups(regions)}
    plans = {region: worker_arguments(region, extra) for region in groups}
    # Pin the exact same patch window for every worker, including year boundaries.
    first = pipeline.parser().parse_args(plans[regions[0]])
    patches = pipeline.resolve_patch_window(first)
    env = dict(os.environ, RIOT_API_KEY=key, PYTHONUNBUFFERED="1")
    root = pipeline.APP_DIR / ".scrape_regions"
    workers, retry_at, finished = {}, {}, {}
    last_status = time.monotonic()
    try:
        with collector_lock(workers):
            while len(finished) < len(groups):
                for region in groups:
                    if region in finished:
                        continue
                    if region in workers:
                        process, stream = workers[region]
                        code = process.poll()
                        if code is None:
                            continue
                        stream.close()
                        if process.stdin is not None:
                            process.stdin.close()
                        del workers[region]
                        if code in (0, 2, 130):
                            finished[region] = code
                            print(f"[{region}] Finished, exit={code}. Log: {root / region / 'collector.log'}", flush=True)
                            continue
                        retry_at[region] = time.monotonic() + 10
                        print(f"[{region}] Process interrupted, restart in 10 seconds.", flush=True)
                    if time.monotonic() < retry_at.get(region, 0):
                        continue
                    folder = root / region
                    folder.mkdir(parents=True, exist_ok=True)
                    stream = (folder / "collector.log").open("a", encoding="utf-8")
                    stream.write(f"\n=== {datetime.now(timezone.utc).isoformat()} ===\n")
                    stream.flush()
                    # Reuse each worker's journal-resolved window on later launches.
                    # The first launch gets the common window via a separate environment variable.
                    child_env = dict(env, ERO_COLLECT_PATCHES=json.dumps(patches))
                    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", region,
                               "--worker-regions", *groups[region], *plans[region]]
                    if region in retry_at and "--resume" not in command:
                        command.append("--resume")
                    try:
                        process = subprocess.Popen(command, cwd=pipeline.APP_DIR, env=child_env,
                            stdin=subprocess.PIPE, stdout=stream, stderr=subprocess.STDOUT,
                            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0)
                    except BaseException:
                        stream.close()
                        raise
                    workers[region] = process, stream
                    print(f"[{region} / {ROUTES[region]}] Started. Log: {folder / 'collector.log'}", flush=True)
                if time.monotonic() - last_status >= 15:
                    for region in regions:
                        try:
                            state = json.loads((root / region / "state.json").read_text(encoding="utf-8"))
                            count = sum(row.get("window_matches", 0) for row in state.get("collection", {}).values())
                            print(f"[{region}] {count} matching stored games in this patch window.", flush=True)
                        except (OSError, ValueError, TypeError):
                            pass
                    last_status = time.monotonic()
                time.sleep(1)
        return max(finished.values(), default=0)
    finally:
        stop_workers(workers)


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--regions", nargs="+", default=["euw1", "eun1", "kr", "na1"], choices=ROUTES)
    cli.add_argument("--key-profile", choices=("default", "development"), default="default")
    cli.add_argument("--worker", choices=ROUTES, help=argparse.SUPPRESS)
    cli.add_argument("--worker-regions", nargs="+", choices=ROUTES, help=argparse.SUPPRESS)
    args, extra = cli.parse_known_args(argv)
    try:
        if args.worker:
            pipeline.REGIONAL_WORKER = args.worker
            pinned = os.environ.get("ERO_COLLECT_PATCHES")
            if not pinned:
                raise ValueError("Regional workers must be started by the collector supervisor")
            if pinned:
                # Parent crash closes the pipe: don't leave orphan collectors outside its lock.
                def watch_parent():
                    sys.stdin.buffer.read()
                    _thread.interrupt_main()
                threading.Thread(target=watch_parent, daemon=True).start()
                pipeline.fetch_versions = lambda: [p + ".1" for p in reversed(json.loads(pinned))]
            return collect_group(args.worker_regions or [args.worker], extra)
        regions = selected_routes(args.regions)
        if "--dry-run" in extra:
            for region in regions:
                print(f"[{region} / {ROUTES[region]}]")
                pipeline.main(worker_arguments(region, extra))
            return 0
        return supervise(regions, args.key_profile, extra)
    except KeyboardInterrupt:
        print("Stopped by user. All region checkpoints kept.")
        return 130
    except (ValueError, pipeline.UpdateLockError) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
