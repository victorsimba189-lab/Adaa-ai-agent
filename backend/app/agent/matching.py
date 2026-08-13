"""
The ADAA workforce matching engine.

This is ordinary Python. There is no AI in this file, and that is the point.

The build specification (section 10) is explicit: Gemini must not do the
mathematics or the geography itself. Gemini reads a request in plain
language and explains the result afterwards, but the decision about who is
eligible and who ranks highest is made here, where it can be tested.

Two things follow from that:

1. Every candidate returned by this module has passed every filter. If a
   worker appears in the result, the database says they are verified,
   skilled, free on the day and close enough to travel.
2. If we cannot fill a request, we say so. We never pad a result to reach
   the requested number.
"""

from dataclasses import dataclass, field
from datetime import date
from math import asin, cos, radians, sin, sqrt

from app.database import fetch_all

# --- Ranking weights, from specification section 10 -----------------------
# These are prototype weights. They are NOT scientifically validated, and
# they are kept here in one place so they are easy to change and to report
# in the dissertation.
WEIGHTS = {
    "skill": 0.30,
    "availability": 0.20,
    "reliability": 0.20,
    "rating": 0.15,
    "proximity": 0.10,
    "experience": 0.05,
}

# Values used to turn raw numbers into a 0-1 score.
YEARS_FOR_FULL_SKILL_SCORE = 10
YEARS_FOR_FULL_EXPERIENCE_SCORE = 15
AVAILABILITY_WINDOW_DAYS = 7
DEFAULT_SEARCH_RADIUS_KM = 25


# ---------------------------------------------------------------------------
# Geography
# ---------------------------------------------------------------------------

def calculate_distance(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """
    Straight-line distance between two points, in kilometres.

    This is the haversine formula, which treats the earth as a sphere. It is
    accurate enough for deciding whether a worker can reach a site, and it
    needs no extra libraries or PostGIS.
    """
    earth_radius_km = 6371.0

    lat1, lng1, lat2, lng2 = map(radians, [lat1, lng1, lat2, lng2])
    delta_lat = lat2 - lat1
    delta_lng = lng2 - lng1

    a = sin(delta_lat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(delta_lng / 2) ** 2
    return round(2 * earth_radius_km * asin(sqrt(a)), 2)


# ---------------------------------------------------------------------------
# What we are looking for, and what we found
# ---------------------------------------------------------------------------

@dataclass
class WorkforceRequest:
    """A contractor's requirement, once it has been turned into structured data."""

    skill: str
    quantity: int
    on_date: date
    location_lat: float
    location_lng: float
    location_name: str = ""
    max_distance_km: float = DEFAULT_SEARCH_RADIUS_KM


@dataclass
class Candidate:
    """
    One worker or one crew that passed every filter.

    ``supply`` is how many people this candidate can contribute:
    1 for an individual worker, or the number of qualified available
    members for a crew.
    """

    kind: str               # "worker" or "crew"
    id: str
    name: str
    supply: int
    distance_km: float
    match_score: float                      # 0-100
    scores: dict = field(default_factory=dict)
    evidence: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "id": self.id,
            "name": self.name,
            "supply": self.supply,
            "distance_km": self.distance_km,
            "match_score": self.match_score,
            "scores": self.scores,
            "evidence": self.evidence,
        }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _capped(value: float | None, maximum: float) -> float:
    """Turn a raw number into a score between 0 and 1."""
    if value is None:
        return 0.0
    return max(0.0, min(float(value) / maximum, 1.0))


def _proximity_score(distance_km: float, max_distance_km: float) -> float:
    """1.0 at the site itself, falling to 0.0 at the edge of the search area."""
    if max_distance_km <= 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - (distance_km / max_distance_km)))


def score_candidate(row: dict, distance_km: float, request: WorkforceRequest) -> tuple[float, dict]:
    """
    Work out how well one candidate fits the request.

    Returns the overall score out of 100, and the six parts it was built
    from, so the agent can explain the ranking honestly.
    """
    scores = {
        # Depth of experience in the SKILL THAT WAS ASKED FOR. Everyone here
        # already has it verified -- the filter guaranteed that -- so this
        # measures how established they are in it.
        "skill": _capped(row.get("skill_years"), YEARS_FOR_FULL_SKILL_SCORE),

        # How much of the week around the job they are free. Someone free
        # for the whole week is less likely to drop out than someone
        # squeezing the job between two others.
        "availability": _capped(row.get("free_days"), AVAILABILITY_WINDOW_DAYS),

        "reliability": _capped(row.get("reliability_score"), 5.0),
        "rating": _capped(row.get("rating"), 5.0),
        "proximity": _proximity_score(distance_km, request.max_distance_km),
        "experience": _capped(row.get("experience_years"),
                              YEARS_FOR_FULL_EXPERIENCE_SCORE),
    }

    total = sum(WEIGHTS[name] * value for name, value in scores.items())

    rounded = {name: round(value, 3) for name, value in scores.items()}
    return round(total * 100, 1), rounded


# ---------------------------------------------------------------------------
# Finding candidates
# ---------------------------------------------------------------------------

# A worker is eligible only if ALL of these hold (specification section 10):
#   the skill matches, and it is verified
#   the worker is verified
#   the worker is free on the day, according to the availability table
#   the worker is not already committed to another job that day
#   the site is within the worker's travel radius
_WORKER_SQL = """
    select w.id, w.name, w.location_name, w.location_lat, w.location_lng,
           w.travel_radius_km, w.experience_years, w.reliability_score,
           w.average_rating as rating, w.completed_jobs, w.attendance_rate,
           ws.years_experience as skill_years,
           (select count(*)
              from availability a2
             where a2.worker_id = w.id
               and a2.status = 'available'
               and a2.date >= %(on_date)s
               and a2.date < %(on_date)s::date + %(window)s::int) as free_days,
           crew.crew_id, crew.crew_name, crew.leader_id, crew.leader_name,
           crew.leader_status
      from workers w
      left join lateral (
           select c.id as crew_id, c.name as crew_name,
                  leader.id as leader_id, leader.name as leader_name,
                  (select a3.status from availability a3
                    where a3.worker_id = leader.id
                      and a3.date = %(on_date)s) as leader_status
             from crew_members cm
             join crews c on c.id = cm.crew_id
             left join workers leader on leader.id = c.leader_worker_id
            where cm.worker_id = w.id and cm.status = 'active'
            order by cm.joined_at desc, c.id
            limit 1
      ) crew on true
      join worker_skills ws on ws.worker_id = w.id
                           and ws.verification_status = 'verified'
      join skills s on s.id = ws.skill_id
                   and lower(s.name) = lower(%(skill)s)
      join availability a on a.worker_id = w.id
                         and a.date = %(on_date)s
                         and a.status = 'available'
     where w.verification_status = 'verified'
       and w.availability_status <> 'unavailable'
       and not exists (
             select 1
               from job_assignments ja
               join jobs j on j.id = ja.job_id
              where ja.worker_id = w.id
                and j.date = %(on_date)s
                and ja.status in ('accepted', 'confirmed'))
"""

_CREW_MEMBER_SQL = """
    select c.id as crew_id, c.name as crew_name, c.primary_trade,
           c.location_name, c.location_lat, c.location_lng,
           c.travel_radius_km, c.rating, c.reliability_score,
           c.completed_jobs, c.verification_status, c.availability_status,
           (select leader.name from workers leader
             where leader.id = c.leader_worker_id) as leader_name,
           (select a3.status from availability a3
             where a3.worker_id = c.leader_worker_id
               and a3.date = %(on_date)s) as leader_status,
           w.id as worker_id, w.name as worker_name,
           w.experience_years, ws.years_experience as skill_years,
           (select count(*)
              from availability a2
             where a2.worker_id = w.id
               and a2.status = 'available'
               and a2.date >= %(on_date)s
               and a2.date < %(on_date)s::date + %(window)s::int) as free_days
      from crews c
      join crew_members cm on cm.crew_id = c.id and cm.status = 'active'
      join workers w on w.id = cm.worker_id
      join worker_skills ws on ws.worker_id = w.id
                           and ws.verification_status = 'verified'
      join skills s on s.id = ws.skill_id
                   and lower(s.name) = lower(%(skill)s)
      join availability a on a.worker_id = w.id
                         and a.date = %(on_date)s
                         and a.status = 'available'
     where c.availability_status <> 'unavailable'
       and c.verification_status = 'verified'
       and w.verification_status = 'verified'
"""


def _query_params(request: WorkforceRequest) -> dict:
    return {
        "skill": request.skill,
        "on_date": request.on_date,
        "window": AVAILABILITY_WINDOW_DAYS,
    }


def find_workers(request: WorkforceRequest) -> list[Candidate]:
    """
    Every individual worker who can genuinely do this job, best first.

    Workers who are further away than they are willing to travel, or
    further than the search radius, are dropped.
    """
    candidates = []

    for row in fetch_all(_WORKER_SQL, _query_params(request)):
        distance = calculate_distance(
            request.location_lat, request.location_lng,
            row["location_lat"], row["location_lng"],
        )

        # Respect both limits: how far we are searching, and how far this
        # particular worker is prepared to travel.
        if distance > request.max_distance_km:
            continue
        if distance > (row["travel_radius_km"] or 0):
            continue

        score, parts = score_candidate(row, distance, request)
        candidates.append(Candidate(
            kind="worker",
            id=row["id"],
            name=row["name"],
            supply=1,
            distance_km=distance,
            match_score=score,
            scores=parts,
            evidence={
                "location": row["location_name"],
                "verified_skill": request.skill,
                "years_in_skill": row["skill_years"],
                "average_rating": _as_float(row["rating"]),
                "completed_jobs": row["completed_jobs"],
                "attendance_rate": _as_float(row["attendance_rate"]),
                "reliability_score": _as_float(row["reliability_score"]),
                "free_days_next_week": row["free_days"],
                # Which crew this worker belongs to, if any. A contractor
                # deals with the crew's leader, so the leader's name and
                # availability on the requested date travel with the worker.
                "crew_id": row["crew_id"],
                "crew_name": row["crew_name"],
                "crew_leader": row["leader_name"],
                "crew_leader_available": (row["leader_status"] == "available"
                                          if row["leader_name"] else None),
                "crew_leader_status": (row["leader_status"] or "no record"
                                       if row["leader_name"] else None),
            },
        ))

    return sorted(candidates, key=_ranking_key)


def find_crews(request: WorkforceRequest) -> list[Candidate]:
    """
    Every crew that can supply at least one qualified, available worker.

    A crew's ``supply`` is the number of its active members who personally
    hold the verified skill and are free that day. It is not the crew's
    headcount, and it is not a promise made by the crew's reputation.
    """
    rows = fetch_all(_CREW_MEMBER_SQL, _query_params(request))

    # Group the member rows by crew.
    crews: dict[str, dict] = {}
    for row in rows:
        crew = crews.setdefault(row["crew_id"], {"info": row, "members": []})
        crew["members"].append(row)

    candidates = []
    for crew_id, crew in crews.items():
        info = crew["info"]
        members = crew["members"]

        distance = calculate_distance(
            request.location_lat, request.location_lng,
            info["location_lat"], info["location_lng"],
        )
        if distance > request.max_distance_km:
            continue
        if distance > (info["travel_radius_km"] or 0):
            continue

        # Score the crew on the crew's own reputation, but on the average
        # depth and availability of the members who will actually turn up.
        average = lambda key: sum(  # noqa: E731 - short and clear here
            float(m[key] or 0) for m in members) / len(members)

        crew_row = {
            "skill_years": average("skill_years"),
            "free_days": average("free_days"),
            "reliability_score": info["reliability_score"],
            "rating": info["rating"],
            "experience_years": average("experience_years"),
        }
        score, parts = score_candidate(crew_row, distance, request)

        candidates.append(Candidate(
            kind="crew",
            id=crew_id,
            name=info["crew_name"],
            supply=len(members),
            distance_km=distance,
            match_score=score,
            scores=parts,
            evidence={
                "location": info["location_name"],
                "primary_trade": info["primary_trade"],
                "crew_leader": info["leader_name"],
                "crew_leader_available": (info["leader_status"] == "available"
                                          if info["leader_name"] else None),
                "crew_leader_status": (info["leader_status"] or "no record"
                                       if info["leader_name"] else None),
                "crew_rating": _as_float(info["rating"]),
                "crew_completed_jobs": info["completed_jobs"],
                "qualified_available_members": len(members),
                "member_ids": [m["worker_id"] for m in members],
                "member_names": [m["worker_name"] for m in members],
            },
        ))

    return sorted(candidates, key=_ranking_key)


def _ranking_key(candidate: Candidate):
    """
    Sort best first, and break ties in a fixed way.

    The tie-break matters: the specification asks whether the agent gives
    the same answer twice for unchanged data (section 23). Sorting by id
    last makes the order completely predictable.
    """
    return (-candidate.match_score, -candidate.supply, candidate.id)


def _as_float(value) -> float | None:
    """Postgres returns numeric columns as Decimal; JSON prefers float."""
    return None if value is None else float(value)


# ---------------------------------------------------------------------------
# Composing a workforce
# ---------------------------------------------------------------------------

def compose_workforce(request: WorkforceRequest) -> dict:
    """
    Put together a workforce for the request.

    The approach follows the specification's example: prefer crews, because
    a crew that already works together is easier to coordinate, then top up
    with individual workers.

    A worker who belongs to a crew we have already chosen is NOT offered
    again as an individual. Counting somebody twice would inflate the
    result, which is exactly the kind of false claim rule 1 forbids.

    If the request cannot be filled, the shortfall is reported plainly.
    """
    crews = find_crews(request)
    workers = find_workers(request)

    selection: list[Candidate] = []
    filled = 0
    used_worker_ids: set[str] = set()
    used_crew_ids: set[str] = set()

    def take_crew(crew: Candidate, contribution: int) -> None:
        nonlocal filled
        member_ids = crew.evidence.get("member_ids", [])
        selection.append(Candidate(
            kind="crew",
            id=crew.id,
            name=crew.name,
            supply=contribution,
            distance_km=crew.distance_km,
            match_score=crew.match_score,
            scores=crew.scores,
            evidence={
                **crew.evidence,
                "contributing": contribution,
                "of_available": len(member_ids),
            },
        ))
        filled += contribution
        used_crew_ids.add(crew.id)
        # Everyone in this crew is now spoken for, whether or not all of
        # them are needed, so none of them can also be picked individually.
        used_worker_ids.update(member_ids)

    def free_workers() -> list[Candidate]:
        return [w for w in workers if w.id not in used_worker_ids]

    # --- Step 1: whole crews that fit inside what is still needed ---
    # A crew that already works together is easier to coordinate, so crews
    # come first. But we only take a crew whose members are ALL needed. We
    # do not break a crew apart just to fill the last position or two --
    # the specification (section 15) fills a remainder with individuals.
    for crew in crews:
        still_needed = request.quantity - filled
        if still_needed <= 0:
            break
        if crew.supply <= still_needed:
            take_crew(crew, crew.supply)

    # --- Step 2: individual workers fill the remainder ---
    for worker in free_workers():
        if filled >= request.quantity:
            break
        selection.append(worker)
        used_worker_ids.add(worker.id)
        filled += 1

    # --- Step 3: only if individuals ran out, take part of another crew ---
    # This is the last resort, so the request is filled if it possibly can
    # be, but a crew is never split unnecessarily.
    for crew in crews:
        still_needed = request.quantity - filled
        if still_needed <= 0:
            break
        if crew.id in used_crew_ids:
            continue
        available_members = [
            member_id for member_id in crew.evidence.get("member_ids", [])
            if member_id not in used_worker_ids
        ]
        if not available_members:
            continue
        take_crew(crew, min(len(available_members), still_needed))

    shortfall = max(0, request.quantity - filled)

    return {
        "request": {
            "skill": request.skill,
            "quantity": request.quantity,
            "date": request.on_date.isoformat(),
            "location": request.location_name,
            "location_lat": request.location_lat,
            "location_lng": request.location_lng,
            "search_radius_km": request.max_distance_km,
        },
        "filled": filled,
        "shortfall": shortfall,
        "complete": shortfall == 0,
        "selection": [c.as_dict() for c in selection],
        "considered": {
            "crews": [c.as_dict() for c in crews],
            "workers": [w.as_dict() for w in workers],
        },
        "weights_used": WEIGHTS,
    }
