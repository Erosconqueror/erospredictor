"""Export champion role frequencies from existing PostgreSQL matches, without training."""

import argparse
import os
from pathlib import Path
import sys

import configs as cfg
from services.role_distribution import build_role_distribution, save_role_distribution, DEFAULT_PATH


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    scope = cli.add_mutually_exclusive_group()
    scope.add_argument("--patches", nargs="+", help="Exact stored patch IDs; default uses configs.ALLOWED_PATCHES")
    scope.add_argument("--all-patches", action="store_true", help="Use all patches already in the database")
    cli.add_argument("--tiers", nargs="+", type=str.upper, choices=cfg.TARGET_TIERS, default=list(cfg.TARGET_TIERS))
    cli.add_argument("--region", default=cfg.REGION)
    cli.add_argument("--output", type=Path, default=DEFAULT_PATH)
    return cli


def read_records(patches, tiers, region, connect=None):
    # Import lazily: --help works without psycopg2, Torch or a Riot client/key.
    if connect is None:
        import psycopg2
        connect = psycopg2.connect
    connection = connect(dbname=cfg.DB_NAME, user=cfg.DB_USER, password=cfg.DB_PASS,
                         host=cfg.DB_HOST, port=cfg.DB_PORT, connect_timeout=10)
    try:
        # Do not construct DataManager(True): it initializes/commits the schema.
        connection.set_session(readonly=True, autocommit=True)
        conditions, params = ["tier = ANY(%s)", "region = %s"], [list(tiers), region]
        if patches is not None:
            conditions.append("patch = ANY(%s)")
            params.append(list(patches))
        with connection.cursor() as cursor:
            cursor.execute("SELECT match_id, region, tier, patch, blue_team, red_team FROM matches WHERE "
                           + " AND ".join(conditions), tuple(params))
            return [{"match_id": row[0], "region": row[1], "tier": row[2], "patch": row[3],
                     "blue_team": row[4], "red_team": row[5]} for row in cursor.fetchall()]
    finally:
        connection.close()


def run(args, log=print):
    patches = None if args.all_patches else args.patches or list(cfg.ALLOWED_PATCHES)
    records = read_records(patches, args.tiers, args.region)
    document = build_role_distribution(records, patches=patches, tiers=args.tiers,
                                       region=args.region, source="database_export")
    if document["scope"]["matches"] == 0:
        raise ValueError("Nincs érvényes, a megadott szűrésnek megfelelő meccs. A meglévő JSON megmaradt.")
    document = save_role_distribution(args.output, document, selected_tiers=args.tiers)
    log(f"Szerepkör-eloszlás elmentve: {args.output} ({document['scope']['matches']} meccs).")
    return 0


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        return run(args)
    except Exception as error:
        message = str(error)
        for secret in (getattr(cfg, "DB_PASS", None), getattr(cfg, "RIOT_API_KEY", None), os.environ.get("RIOT_API_KEY")):
            if secret:
                message = message.replace(str(secret), "[redacted]")
        print(f"A szerepkör-export nem fejeződött be: {message}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
