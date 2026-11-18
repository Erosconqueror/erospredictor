"""Transactional PostgreSQL sufficient statistics, shared by every collector."""

import json
from psycopg2.extras import execute_values

SCHEMA = """
CREATE TABLE IF NOT EXISTS champion_trait_aggregates (
    version INTEGER NOT NULL DEFAULT 1,
    patch TEXT NOT NULL, region TEXT NOT NULL, tier TEXT NOT NULL,
    champion INTEGER NOT NULL, role INTEGER NOT NULL CHECK (role BETWEEN 0 AND 4),
    metrics JSONB NOT NULL,
    PRIMARY KEY (version, patch, region, tier, champion, role)
);
"""

UPSERT = """
INSERT INTO champion_trait_aggregates AS stored
    (patch, region, tier, champion, role, metrics) VALUES %s
ON CONFLICT (version, patch, region, tier, champion, role) DO UPDATE SET metrics = (
    SELECT jsonb_object_agg(key, jsonb_build_array(total, samples)) FROM (
        SELECT key, SUM((value->>0)::double precision) AS total,
               SUM((value->>1)::bigint) AS samples
        FROM (SELECT * FROM jsonb_each(stored.metrics)
              UNION ALL SELECT * FROM jsonb_each(EXCLUDED.metrics)) AS observations
        GROUP BY key
    ) AS combined
);
"""


def save_observations(cursor, data, region):
    from services.champion_traits import finite
    rows, seen = [], set()
    for observation in data.get("trait_observations", []):
        champion, role = observation.get("champion"), observation.get("role")
        if (type(champion) is not int or champion <= 0 or type(role) is not int or not 0 <= role < 5
                or (champion, role) in seen):
            continue
        metrics = {key: value for key, value in observation.get("metrics", {}).items()
                   if isinstance(value, (list, tuple)) and len(value) == 2
                   and finite(value[0]) and type(value[1]) is int and value[1] == 1}
        if metrics:
            seen.add((champion, role))
            rows.append((data["patch"], region, data["tier"], champion, role,
                         json.dumps(metrics, allow_nan=False)))
    if rows:
        rows.sort(key=lambda row: row[:5])  # Stable lock order across parallel collectors.
        execute_values(cursor, UPSERT, rows, template="(%s,%s,%s,%s,%s,%s::jsonb)")


def read_observations(connection, patches, tiers=None, region=None):
    conditions, params = ["version = 1", "patch = ANY(%s)"], [list(patches)]
    if tiers is not None:
        conditions.append("tier = ANY(%s)")
        params.append(list(tiers))
    if region is not None:
        conditions.append("region = %s")
        params.append(region)
    with connection.cursor() as cursor:
        cursor.execute("SELECT patch, region, tier, champion, role, metrics FROM champion_trait_aggregates WHERE "
                       + " AND ".join(conditions) + " ORDER BY patch, region, tier, champion, role", params)
        return [dict(zip(("patch", "region", "tier", "champion", "role", "metrics"), row)) for row in cursor.fetchall()]
