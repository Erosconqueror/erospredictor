-- PostgreSQL. Edit this patch list to choose the examined window.
-- Team array order: Top, Jungle, Mid, ADC, Support (Riot champion IDs).
-- Only an entire blue/red team swap is ignored; lane order is preserved.
WITH scoped AS (
    SELECT match_id, patch,
           LEAST(blue_team, red_team) AS team_a,
           GREATEST(blue_team, red_team) AS team_b
    FROM matches
    WHERE patch = ANY(ARRAY['16.18', '16.19', '16.20', '16.21', '16.22']::text[])
      AND array_ndims(blue_team) = 1 AND array_ndims(red_team) = 1
      AND cardinality(blue_team) = 5 AND cardinality(red_team) = 5
      AND 0 < ALL(blue_team) AND 0 < ALL(red_team)
)
SELECT team_a, team_b,
       COUNT(*) AS occurrences,
       ARRAY_AGG(match_id ORDER BY match_id) AS match_ids,
       ARRAY_AGG(DISTINCT patch ORDER BY patch) AS patches
FROM scoped
GROUP BY team_a, team_b
HAVING COUNT(*) > 1
ORDER BY occurrences DESC, team_a, team_b;
