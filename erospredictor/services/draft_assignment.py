"""Assign champion roles jointly from observed role frequencies.

The client is authoritative for the own team's reported roles. Enemy roles are
estimates from P(role | champion), never from the roster's display/cell order.
Every team needs at most 5! role permutations. Unsupported roles stay unresolved
until a complete enemy roster permits an explicitly marked forced placement.
"""

from dataclasses import dataclass, replace
from itertools import permutations
import math

from services.draft_import import DraftPick
from services.role_distribution import role_probabilities

MIN_ROLE_PROBABILITY = .05


@dataclass(frozen=True)
class DraftScenario:
    blue: tuple
    red: tuple
    weight: float
    forced: bool = False


@dataclass(frozen=True)
class PickAssignment:
    team: str
    internal_id: int
    riot_id: int
    role: int | None
    source: str
    role_probability: float | None
    sample_size: int
    probabilities: tuple[float, ...]


@dataclass(frozen=True)
class DraftAssignment:
    values: dict
    assignments: tuple[PickAssignment, ...]
    unresolved: tuple[DraftPick, ...]
    warnings: tuple[str, ...]
    scenarios: tuple[DraftScenario, ...] = ()

    @property
    def complete(self):
        return not self.unresolved


def _index(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _role(value):
    return _index(value) and value < 5


def _bans(snapshot, own):
    relative = snapshot.own_bans or snapshot.enemy_bans
    if relative:
        own_bans, enemy_bans = list(snapshot.own_bans), list(snapshot.enemy_bans)
        blue, red = (own_bans, enemy_bans) if own == "blue" else (enemy_bans, own_bans)
        if ((snapshot.blue_bans and list(snapshot.blue_bans) != blue)
                or (snapshot.red_bans and list(snapshot.red_bans) != red)):
            raise ValueError("A relatív és abszolút tiltáslisták ellentmondanak egymásnak.")
    else:
        blue, red = list(snapshot.blue_bans), list(snapshot.red_bans)
    for values in (blue, red):
        if len(values) > 5 or any(not _index(value) for value in values) or len(set(values)) != len(values):
            raise ValueError("Egy csapat tiltáslistájában legfeljebb öt különböző, érvényes hős szerepelhet.")
    return blue, red


def assign_draft(snapshot, role_distributions, own_side=None, previous_roles=None, tier="MIXED"):
    """Return a draft plus per-pick evidence without changing the snapshot.

    ``previous_roles`` contains only manually confirmed assignments, keyed by
    ``(absolute_team, internal_champion_id)``. A reported own-team role always
    takes precedence. When no role data is available, an explicit manual choice
    is required for that pick and ``unresolved`` records the omitted champion.

    Automatic roles need at least 5% of that champion's observations. Only a
    fully visible five-enemy roster may relax this rule; those placements are
    explicitly marked forced. Keep up to three likely enemy configurations.
    """
    if snapshot.own_team not in (None, "blue", "red") or own_side not in (None, "blue", "red"):
        raise ValueError("A saját csapat oldala csak Kék vagy Piros lehet.")
    if snapshot.own_team and own_side and snapshot.own_team != own_side:
        raise ValueError("A megadott oldal ellentmond a kliens által közölt saját csapatnak.")
    own = snapshot.own_team or own_side
    if own is None:
        raise ValueError("A saját csapat oldala nem biztos. Válaszd ki a Kék vagy Piros oldalt.")
    other = "red" if own == "blue" else "blue"
    previous_roles = previous_roles or {}
    if not isinstance(previous_roles, dict):
        raise ValueError("A kézzel megerősített pozíciók formátuma érvénytelen.")
    blue_bans, red_bans = _bans(snapshot, own)
    values = {"blue": [None] * 5, "red": [None] * 5,
              "blue_bans": blue_bans, "red_bans": red_bans}
    notes = list(snapshot.warnings)
    picks, seen_ids, seen_riot = [], set(), set()
    for pick in snapshot.picks:
        team = own if pick.team == "own" else other if pick.team == "enemy" else pick.team
        if team not in ("blue", "red"):
            raise ValueError("A beolvasott hős csapata nem határozható meg.")
        if not _index(pick.internal_id) or not _index(pick.riot_id) or pick.riot_id == 0:
            raise ValueError("A beolvasott draft érvénytelen hősazonosítót tartalmaz.")
        if pick.internal_id in seen_ids or pick.riot_id in seen_riot:
            raise ValueError("Ugyanaz a hős nem szerepelhet kétszer a draftban.")
        if pick.internal_id in blue_bans + red_bans:
            raise ValueError("Kiválasztott hős nem szerepelhet a tiltások között.")
        seen_ids.add(pick.internal_id)
        seen_riot.add(pick.riot_id)
        picks.append(replace(pick, team=team))
    if any(sum(pick.team == team for pick in picks) > 5 for team in ("blue", "red")):
        raise ValueError("Egy csapatban legfeljebb öt kiválasztott hős szerepelhet.")

    metadata, enemy_plans = {}, []
    for team in ("blue", "red"):
        pending, occupied = [], {}
        team_picks = sorted((pick for pick in picks if pick.team == team),
                            key=lambda pick: (not (team == own and pick.role is not None), pick.internal_id))
        for pick in team_picks:
            probabilities, total = role_probabilities(role_distributions, pick.riot_id, tier=tier)
            probabilities = tuple(probabilities)
            role, source = None, "unresolved"
            manual = previous_roles.get((team, pick.internal_id))
            if manual is not None and not _role(manual):
                raise ValueError("A kézzel megerősített pozíció érvénytelen.")
            if team == own and pick.role is not None:
                if not _role(pick.role):
                    raise ValueError("A kliens érvénytelen saját csapatszerepkört közölt.")
                role, source = pick.role, "client"
                if manual is not None and manual != role:
                    notes.append(f"A(z) {pick.riot_id} hősnél a kliens szerepköre felülírta a korábbi kézi pozíciót.")
            elif manual is not None:
                occupied_id = occupied.get(manual)
                prior = metadata.get((team, occupied_id))
                if prior is not None and prior.source == "client":
                    notes.append(f"A(z) {pick.riot_id} hős korábbi kézi pozícióját a kliens egy csapattárshoz rendelte; új szerepkör szükséges.")
                else:
                    role, source = manual, "manual"
            if role is not None:
                if role in occupied:
                    raise ValueError("A saját vagy kézzel rögzített pozíciók ütköznek: egy szerepkörhöz két hős tartozna.")
                occupied[role] = pick.internal_id
                values[team][role] = pick.internal_id
                probability = probabilities[role] if total > 0 else None
                metadata[(team, pick.internal_id)] = PickAssignment(team, pick.internal_id, pick.riot_id,
                    role, source, probability, total, probabilities if total else ())
            elif total > 0 or (team != own and len(team_picks) == 5):
                pending.append((pick, probabilities, total))
            else:
                metadata[(team, pick.internal_id)] = PickAssignment(team, pick.internal_id, pick.riot_id,
                    None, "unresolved", None, 0, ())
                notes.append(f"A(z) {pick.riot_id} hőshöz nincs szerepkörstatisztika; kézi pozíció szükséges.")

        free = [role for role in range(5) if role not in occupied]
        fallback = team != own and len(team_picks) == 5
        plans = []
        for roles in permutations(free, len(pending)):
            supported = [(probabilities[role], total) for (_, probabilities, total), role in zip(pending, roles)
                         if total and probabilities[role] >= MIN_ROLE_PROBABILITY]
            log_probability = (sum(math.log(max(probabilities[role], 1e-9)) for (_, probabilities, _), role in zip(pending, roles))
                               if fallback else sum(math.log(probability) for probability, _ in supported))
            score = (len(supported), log_probability,
                     sum(math.log1p(total) for _, total in supported))
            plans.append((score, roles))
        plans.sort(key=lambda item: item[0], reverse=True)
        best_score, best = plans[0] if plans else ((0, 0, 0), ())
        ties = sum(score == best_score for score, _ in plans)
        if ties > 1 and pending:
            notes.append(f"A {'Kék' if team == 'blue' else 'Piros'} csapat több hasonló szerepkör-felállással is összeegyeztethető; a becslés kézzel javítható.")
        for (pick, probabilities, total), role in zip(pending, best or ()):
            probability = probabilities[role]
            supported = total > 0 and probability >= MIN_ROLE_PROBABILITY
            if supported or fallback:
                values[team][role] = pick.internal_id
                metadata[(team, pick.internal_id)] = PickAssignment(team, pick.internal_id, pick.riot_id,
                    role, "statistics" if supported else "forced", probability if total else None,
                    total, probabilities if total else ())
                if not supported:
                    notes.append(f"A(z) {pick.riot_id} hős kényszerített pozíciót kapott: az összes ellenfél látható, de nincs teljes, legalább 5%-os kiosztás. Ellenőrizd kézzel.")
                elif total < 20:
                    notes.append(f"A(z) {pick.riot_id} hős szerepköre kis mintából ({total} választás) becsült.")
            else:
                metadata[(team, pick.internal_id)] = PickAssignment(team, pick.internal_id, pick.riot_id,
                    None, "unresolved", None, total, probabilities)
                notes.append(f"A(z) {pick.riot_id} hőshöz nincs ütközésmentes, legalább 5%-os szerepkör; kézi pozíció szükséges.")

        if team != own:
            seen_lineups = set()
            for score, roles in plans:
                if score[0] != best_score[0]:
                    continue
                lineup = [None] * 5
                for role, cid in occupied.items():
                    lineup[role] = cid
                forced = False
                for (pick, probabilities, total), role in zip(pending, roles):
                    supported = total > 0 and probabilities[role] >= MIN_ROLE_PROBABILITY
                    if supported or fallback:
                        lineup[role] = pick.internal_id
                        forced |= not supported
                signature = tuple(lineup)
                if signature in seen_lineups:
                    continue
                seen_lineups.add(signature)
                enemy_plans.append((signature, math.exp(score[1] - best_score[1]), forced))
                if len(enemy_plans) == 3:
                    break

    assignments = tuple(metadata[(pick.team, pick.internal_id)] for pick in picks)
    unresolved = tuple(replace(pick, role=None) for pick, assignment in zip(picks, assignments)
                       if assignment.role is None)
    total_weight = sum(weight for _, weight, _ in enemy_plans)
    scenarios = tuple(DraftScenario(tuple(values["blue"]) if own == "blue" else lineup,
                                   lineup if own == "blue" else tuple(values["red"]), weight / total_weight, forced)
                      for lineup, weight, forced in enemy_plans)
    return DraftAssignment(values, assignments, unresolved, tuple(dict.fromkeys(notes)), scenarios)
