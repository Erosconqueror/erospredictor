"""Assign champion roles jointly from observed role frequencies.

The client is authoritative for the own team's reported roles. Enemy roles are
estimates from P(role | champion), never from the roster's display/cell order.
Every team needs at most 5! role permutations; unknown or unobserved roles stay
unresolved rather than gaining an invented probability.
"""

from dataclasses import dataclass, replace
from itertools import permutations
import math

from services.draft_import import DraftPick
from services.role_distribution import role_probabilities


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

    With observed positive frequencies, maximize the joint product subject to
    unique roles. If a full matching is impossible, maximize the number of
    supported placements first, then the positive-frequency product. An unused
    empirical 0% role never becomes an automatically accepted placement.
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

    metadata = {}
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
            elif total > 0:
                pending.append((pick, probabilities, total))
            else:
                metadata[(team, pick.internal_id)] = PickAssignment(team, pick.internal_id, pick.riot_id,
                    None, "unresolved", None, 0, ())
                notes.append(f"A(z) {pick.riot_id} hőshöz nincs szerepkörstatisztika; kézi pozíció szükséges.")

        free = [role for role in range(5) if role not in occupied]
        best, best_score, ties = None, None, 0
        for roles in permutations(free, len(pending)):
            supported = [(probabilities[role], total) for (_, probabilities, total), role in zip(pending, roles)
                         if probabilities[role] > 0]
            score = (len(supported), sum(math.log(probability) for probability, _ in supported),
                     sum(math.log1p(total) for _, total in supported))
            if best_score is None or score > best_score:
                best, best_score, ties = roles, score, 1
            elif score == best_score:
                ties += 1
        if ties > 1 and pending:
            notes.append(f"A {'Kék' if team == 'blue' else 'Piros'} csapat több hasonló szerepkör-felállással is összeegyeztethető; a becslés kézzel javítható.")
        for (pick, probabilities, total), role in zip(pending, best or ()):
            probability = probabilities[role]
            if probability > 0:
                values[team][role] = pick.internal_id
                metadata[(team, pick.internal_id)] = PickAssignment(team, pick.internal_id, pick.riot_id,
                    role, "statistics", probability, total, probabilities)
                if total < 20:
                    notes.append(f"A(z) {pick.riot_id} hős szerepköre kis mintából ({total} választás) becsült.")
            else:
                metadata[(team, pick.internal_id)] = PickAssignment(team, pick.internal_id, pick.riot_id,
                    None, "unresolved", None, total, probabilities)
                notes.append(f"A(z) {pick.riot_id} hőshöz nincs ütközésmentes, megfigyelt szerepkör; kézi pozíció szükséges.")

    assignments = tuple(metadata[(pick.team, pick.internal_id)] for pick in picks)
    unresolved = tuple(replace(pick, role=None) for pick, assignment in zip(picks, assignments)
                       if assignment.role is None)
    return DraftAssignment(values, assignments, unresolved, tuple(dict.fromkeys(notes)))
