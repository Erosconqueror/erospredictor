"""Developer-only match retention. Preview by default; --apply backs up and deletes."""

import argparse
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import tempfile
import uuid

import configs as cfg
from update_pipeline import fetch_versions, normalize_patch, positive_int

BACKUP_DIRECTORY = Path(__file__).resolve().parent / ".database_maintenance" / "backups"
COLUMNS = ("match_id", "region", "tier", "patch", "blue_win", "blue_team", "red_team", "created_at")


def retention_plan(counts, versions, keep=26):
    if not isinstance(keep, int) or isinstance(keep, bool) or keep < 1:
        raise ValueError("Legalább egy patch megőrzése kötelező.")
    released = set()
    for version in versions:
        try:
            released.add(normalize_patch(version))
        except (ValueError, AttributeError):
            continue
    released = sorted(released, key=lambda patch: tuple(map(int, patch.split("."))))
    if len(released) < keep:
        raise ValueError("Nincs elegendő igazoltan megjelent patch; törlés nem történhet.")
    kept = released[-keep:]
    cutoff = tuple(map(int, kept[0].split(".")))
    old, unknown = [], []
    total = 0
    for patch, count in counts:
        total += count
        try:
            key = tuple(map(int, normalize_patch(patch).split(".")))
        except (ValueError, AttributeError):
            unknown.append({"patch": patch, "matches": count})
            continue
        if key < cutoff:
            old.append({"patch": patch, "matches": count})
    return {"keep_patch_count": keep, "kept_patches": kept, "cutoff_inclusive": kept[0],
            "latest_released": kept[-1], "old_patches": old, "preserved_unknown": unknown,
            "candidate_count": sum(row["matches"] for row in old), "total_matches": total}


def write_backup(rows, plan, directory):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = directory / f"matches-before-retention-{stamp}-{uuid.uuid4().hex}.json.gz"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=directory, prefix=".pending-", delete=False) as output:
            temporary = Path(output.name)
        payload = {"version": 1, "kind": "erospredictor_match_retention_backup", "plan": plan,
                   "row_count": len(rows), "columns": list(COLUMNS), "rows": rows}
        with gzip.open(temporary, "wt", encoding="utf-8") as output:
            json.dump(payload, output, ensure_ascii=False,
                      default=lambda value: value.isoformat() if isinstance(value, datetime) else str(value))
        with temporary.open("rb+") as output:
            os.fsync(output.fileno())
        # Verify the compressed archive before allowing any destructive SQL.
        with gzip.open(temporary, "rt", encoding="utf-8") as archive:
            verified = json.load(archive)
        if (verified["row_count"] != len(rows) or len(verified["rows"]) != len(rows)
                or [row[0] for row in verified["rows"]] != [row[0] for row in rows]):
            raise ValueError("A biztonsági mentés ellenőrzése sikertelen; nincs törlés.")
        os.replace(temporary, destination)
        temporary = None
        return destination
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def maintain(connection, versions, *, keep=26, apply=False, region=None, backup_directory=BACKUP_DIRECTORY):
    """Only exact IDs of backed-up, locked old rows are deleted in one transaction."""
    connection.set_session(readonly=not apply, autocommit=False)
    scope = " WHERE region = %s" if region is not None else ""
    params = (region,) if region is not None else ()
    try:
        with connection.cursor() as cursor:
            cursor.execute("SET LOCAL lock_timeout = '5s'")
            cursor.execute("SET LOCAL statement_timeout = '120s'")
            cursor.execute("SELECT patch, COUNT(*) FROM matches" + scope + " GROUP BY patch ORDER BY patch", params)
            plan = retention_plan(cursor.fetchall(), versions, keep)
            plan["region"] = region
            report = {"mode": "apply" if apply else "preview", "plan": plan, "deleted": 0, "backup": None}
            if not apply or not plan["old_patches"]:
                connection.rollback()
                return report
            old_patches = [row["patch"] for row in plan["old_patches"]]
            where = "patch = ANY(%s)" + (" AND region = %s" if region is not None else "")
            selection = (old_patches, region) if region is not None else (old_patches,)
            cursor.execute("SELECT " + ", ".join(COLUMNS) + " FROM matches WHERE " + where + " FOR UPDATE", selection)
            rows = cursor.fetchall()
            if not rows:
                connection.rollback()
                return report
            backup = write_backup(rows, plan, backup_directory)
            ids = [row[0] for row in rows]
            deleted = 0
            for start in range(0, len(ids), 5000):
                cursor.execute("DELETE FROM matches WHERE match_id = ANY(%s) AND patch = ANY(%s)",
                               (ids[start:start + 5000], old_patches))
                deleted += cursor.rowcount
            if deleted != len(rows):
                raise ValueError("A törlendő sorok száma eltér a mentéstől; a tranzakció visszavonva.")
            connection.commit()
            report.update(deleted=deleted, backup=str(backup))
            return report
    except BaseException:
        connection.rollback()
        raise


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--keep-patches", type=positive_int, default=26, help="Number of latest released patches to retain (default: 26)")
    cli.add_argument("--apply", action="store_true", help="Back up and delete rows older than the retention window")
    cli.add_argument("--region", help="Optional exact database region; default covers all regions")
    cli.add_argument("--backup-directory", type=Path, default=BACKUP_DIRECTORY)
    return cli


def main(argv=None):
    args = parser().parse_args(argv)
    connection = None
    try:
        versions = fetch_versions()
        import psycopg2
        connection = psycopg2.connect(dbname=cfg.DB_NAME, user=cfg.DB_USER, password=cfg.DB_PASS,
                                     host=cfg.DB_HOST, port=cfg.DB_PORT, connect_timeout=10)
        report = maintain(connection, versions, keep=args.keep_patches, apply=args.apply,
                          region=args.region, backup_directory=args.backup_directory)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except KeyboardInterrupt:
        print("A karbantartás megszakítva; a nem véglegesített törlés visszavonva.")
        return 130
    except Exception as error:
        message = str(error)
        for secret in (getattr(cfg, "DB_PASS", None), getattr(cfg, "RIOT_API_KEY", None), os.environ.get("RIOT_API_KEY")):
            if secret:
                message = message.replace(str(secret), "[redacted]")
        print("Az adatbázis-karbantartás sikertelen: " + message)
        return 2
    finally:
        if connection is not None:
            connection.close()


if __name__ == "__main__":
    raise SystemExit(main())
