"""
The tools Gemini is allowed to use.

This is the only way the agent can reach ADAA's data. Every function here
reads from PostgreSQL and returns what it found. None of them let the model
decide who is eligible -- that is still the matching engine's job, and
these tools call it.

Why the agent gets tools instead of being told the data
-------------------------------------------------------
A language model asked about eight masons in Guntur will produce eight
plausible masons whether or not any exist. Giving it tools does not make it
honest by itself, but it means every worker it mentions came from a
database row, and we can prove which one.

Three habits are kept throughout:

1. A tool never returns an opinion. It returns records, plus the numbers
   behind them, so the agent explains rather than asserts.
2. A tool that finds nothing says so plainly. "No available verified masons
   in Guntur tomorrow" is a useful answer. An invented one is not.
3. The tools that would change something do not change anything. They
   write down a proposal and return its id; a person confirms it through
   the API. There is no tool for confirming, on purpose -- see
   actions.py.
"""

from datetime import date, timedelta
from decimal import Decimal

from app.agent.matching import (
    WorkforceRequest,
    calculate_distance,
    compose_workforce,
    find_crews,
    find_workers,
)
from app.agent import actions, audit, independence
from app.agent.audit import logged
from app.database import all_locations, fetch_all, fetch_one, find_location


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _plain(value):
    """
    Convert database types into things that survive being turned into JSON.

    PostgreSQL returns numeric columns as Decimal and dates as date
    objects. Neither can be sent to Gemini as-is.
    """
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


def _resolve_date(on_date: str) -> date:
    """Blank means tomorrow, which is what most requests are about."""
    if not on_date:
        return date.today() + timedelta(days=1)
    try:
        return date.fromisoformat(on_date)
    except ValueError:
        return date.today() + timedelta(days=1)


def _place(location: str) -> dict | None:
    return find_location(location) if location else None


def _unknown_place(location: str) -> dict:
    known = [row["name"] for row in all_locations()]
    return {
        "found": 0,
        "results": [],
        "note": (
            f"ADAA has no workforce recorded at '{location}'. "
            f"Places with workers: {', '.join(known)}."
        ),
    }


# ---------------------------------------------------------------------------
# Tool 1 - search_workers
# ---------------------------------------------------------------------------

def search_workers(skill: str, location: str, on_date: str = "",
                   radius_km: float = 25.0, limit: int = 10) -> dict:
    """
    Find individual workers who can genuinely do a job.

    Every worker returned holds the skill as a VERIFIED skill, is verified
    themselves, is free on that date according to the availability table,
    is not already committed to another job, and is close enough to travel.
    Anyone failing any of those is not returned at all.

    Each result also says which crew the worker belongs to (crew_name),
    who leads that crew (crew_leader) and whether that leader is free on
    the date (crew_leader_available). A worker with no crew has null in
    those fields and works independently.

    Args:
        skill: the trade needed, for example "Mason" or "Carpenter".
        location: the place name, for example "Guntur".
        on_date: the date as YYYY-MM-DD. Leave blank for tomorrow.
        radius_km: how far from the site to search.
        limit: how many to return, best match first.
    """
    place = _place(location)
    if place is None:
        return _unknown_place(location)

    wanted = _resolve_date(on_date)
    request = WorkforceRequest(
        skill=skill, quantity=limit, on_date=wanted,
        location_lat=place["lat"], location_lng=place["lng"],
        location_name=place["name"], max_distance_km=radius_km,
    )

    found = find_workers(request)[:limit]

    return _plain({
        "searched_for": {"skill": skill, "location": place["name"],
                         "date": wanted.isoformat(), "radius_km": radius_km},
        "found": len(found),
        "results": [{
            "worker_id": worker.id,
            "name": worker.name,
            "distance_km": worker.distance_km,
            "match_score": worker.match_score,
            **worker.evidence,
        } for worker in found],
        "note": ("No verified workers with that skill are available there on "
                 "that date." if not found else
                 "All of these are verified, skilled and free on the date. "
                 "crew_name and crew_leader say which crew each worker "
                 "belongs to and who leads it; crew_leader_available is "
                 "whether that leader is free on the date. Nulls mean the "
                 "worker is independent."),
    })


# ---------------------------------------------------------------------------
# Tool 2 - search_crews
# ---------------------------------------------------------------------------

def search_crews(skill: str, location: str, on_date: str = "",
                 radius_km: float = 25.0, limit: int = 10) -> dict:
    """
    Find crews that can supply workers for a job.

    A crew's "available_workers" is the number of its active members who
    personally hold the verified skill and are free that day. It is not the
    crew's headcount, and it is never guessed from the crew's rating.

    Args:
        skill: the trade needed, for example "Mason".
        location: the place name, for example "Guntur".
        on_date: the date as YYYY-MM-DD. Leave blank for tomorrow.
        radius_km: how far from the site to search.
        limit: how many crews to return, best match first.
    """
    place = _place(location)
    if place is None:
        return _unknown_place(location)

    wanted = _resolve_date(on_date)
    request = WorkforceRequest(
        skill=skill, quantity=limit, on_date=wanted,
        location_lat=place["lat"], location_lng=place["lng"],
        location_name=place["name"], max_distance_km=radius_km,
    )

    found = find_crews(request)[:limit]

    return _plain({
        "searched_for": {"skill": skill, "location": place["name"],
                         "date": wanted.isoformat(), "radius_km": radius_km},
        "found": len(found),
        "results": [{
            "crew_id": crew.id,
            "name": crew.name,
            "available_workers": crew.supply,
            "distance_km": crew.distance_km,
            "match_score": crew.match_score,
            **crew.evidence,
        } for crew in found],
        "note": ("No crews with that skill are available there on that date."
                 if not found else
                 "available_workers counts members who personally hold the "
                 "verified skill and are free that day."),
    })


# ---------------------------------------------------------------------------
# Tool 3 - get_worker_profile
# ---------------------------------------------------------------------------

def get_worker_profile(worker_id: str) -> dict:
    """
    Everything ADAA knows about one worker.

    Includes their crew history. A worker who has left a crew keeps their
    jobs, ratings and verified skills -- the membership record simply gets
    an end date. Their reputation belongs to them, not to the crew.

    Args:
        worker_id: the worker's ADAA id, for example "W001".
    """
    worker = fetch_one(
        """
        select id, name, location_name, preferred_language, travel_radius_km,
               experience_years, verification_status, availability_status,
               reliability_score, average_rating, completed_jobs,
               attendance_rate
          from workers where id = %s
        """,
        (worker_id,),
    )
    if worker is None:
        return {"found": False,
                "note": f"There is no worker with id {worker_id}."}

    worker["skills"] = fetch_all(
        """
        select s.name, ws.verification_status, ws.years_experience
          from worker_skills ws join skills s on s.id = ws.skill_id
         where ws.worker_id = %s order by s.name
        """,
        (worker_id,),
    )
    worker["crew_history"] = fetch_all(
        """
        select c.id as crew_id, c.name as crew_name, cm.role, cm.status,
               cm.joined_at, cm.left_at
          from crew_members cm join crews c on c.id = cm.crew_id
         where cm.worker_id = %s order by cm.joined_at desc
        """,
        (worker_id,),
    )
    worker["recent_ratings"] = fetch_all(
        """
        select r.rating, r.comment, j.title, j.date
          from ratings r join jobs j on j.id = r.job_id
         where r.worker_id = %s order by j.date desc limit 5
        """,
        (worker_id,),
    )
    worker["contractors_worked_for"] = fetch_all(
        """
        select c.company_name, count(*) as jobs
          from job_assignments ja
          join jobs j on j.id = ja.job_id
          join contractors c on c.id = j.contractor_id
         where ja.worker_id = %s and ja.status = 'completed'
         group by c.company_name order by count(*) desc
        """,
        (worker_id,),
    )

    return _plain({"found": True, **worker})


# ---------------------------------------------------------------------------
# Tool 4 - get_crew_profile
# ---------------------------------------------------------------------------

def get_crew_profile(crew_id: str) -> dict:
    """
    Everything ADAA knows about one crew, including its members.

    The crew's rating and each member's own rating are reported separately,
    because they are separate things. A crew rated 4.8 does not make every
    member a 4.8 worker.

    Args:
        crew_id: the crew's ADAA id, for example "RAVI01".
    """
    crew = fetch_one(
        """
        select c.id, c.name, c.primary_trade, c.location_name,
               c.availability_status, c.rating as crew_rating,
               c.completed_jobs as crew_completed_jobs,
               c.reliability_score, c.verification_status,
               leader.name as leader_name
          from crews c left join workers leader on leader.id = c.leader_worker_id
         where c.id = %s
        """,
        (crew_id,),
    )
    if crew is None:
        return {"found": False, "note": f"There is no crew with id {crew_id}."}

    crew["members"] = fetch_all(
        """
        select w.id as worker_id, w.name, cm.role, cm.status,
               cm.joined_at, cm.left_at,
               w.average_rating as this_workers_own_rating,
               w.completed_jobs as this_workers_own_completed_jobs,
               w.availability_status
          from crew_members cm join workers w on w.id = cm.worker_id
         where cm.crew_id = %s
         order by (cm.role = 'leader') desc, w.name
        """,
        (crew_id,),
    )

    return _plain({
        "found": True,
        **crew,
        "note": ("crew_rating belongs to the crew. Each member's own rating "
                 "is listed separately and is theirs."),
    })


# ---------------------------------------------------------------------------
# Tool 5 - check_availability
# ---------------------------------------------------------------------------

def check_availability(worker_id: str = "", crew_id: str = "",
                       on_date: str = "") -> dict:
    """
    Check whether a worker or a crew is actually free on a date.

    This reads the availability table, which is the only thing that decides
    the answer. Never state that somebody is available without checking
    here first.

    Args:
        worker_id: a worker id such as "W001". Leave blank if checking a crew.
        crew_id: a crew id such as "RAVI01". Leave blank if checking a worker.
        on_date: the date as YYYY-MM-DD. Leave blank for tomorrow.
    """
    wanted = _resolve_date(on_date)

    if worker_id:
        row = fetch_one(
            """
            select w.name, a.status, a.start_time, a.end_time
              from workers w
              left join availability a on a.worker_id = w.id and a.date = %s
             where w.id = %s
            """,
            (wanted, worker_id),
        )
        if row is None:
            return {"found": False,
                    "note": f"There is no worker with id {worker_id}."}

        status = row["status"]
        return _plain({
            "found": True, "worker_id": worker_id, "name": row["name"],
            "date": wanted.isoformat(),
            "available": status == "available",
            "status": status or "no record for that date",
            "hours": (f"{row['start_time']} to {row['end_time']}"
                      if status == "available" else None),
        })

    if crew_id:
        crew = fetch_one("select name from crews where id = %s", (crew_id,))
        if crew is None:
            return {"found": False,
                    "note": f"There is no crew with id {crew_id}."}

        members = fetch_all(
            """
            select w.id as worker_id, w.name,
                   coalesce(a.status, 'unknown') as status
              from crew_members cm
              join workers w on w.id = cm.worker_id
              left join availability a on a.worker_id = w.id and a.date = %s
             where cm.crew_id = %s and cm.status = 'active'
             order by w.name
            """,
            (wanted, crew_id),
        )
        free = [m for m in members if m["status"] == "available"]

        return _plain({
            "found": True, "crew_id": crew_id, "name": crew["name"],
            "date": wanted.isoformat(),
            "active_members": len(members),
            "available_members": len(free),
            "members": members,
            "note": ("This counts members who are free. It does not check "
                     "whether they hold a particular skill -- use "
                     "search_crews for that."),
        })

    return {"found": False,
            "note": "Give either a worker_id or a crew_id."}


# ---------------------------------------------------------------------------
# Tool 6 - calculate_distance
# ---------------------------------------------------------------------------

def distance_between(from_location: str, to_location: str) -> dict:
    """
    Distance in kilometres between two places ADAA knows about.

    Args:
        from_location: a place name, for example "Guntur".
        to_location: a place name, for example "Vijayawada".
    """
    start = _place(from_location)
    end = _place(to_location)

    missing = [name for name, value in
               [(from_location, start), (to_location, end)] if value is None]
    if missing:
        known = [row["name"] for row in all_locations()]
        return {"found": False,
                "note": (f"Not a known place: {', '.join(missing)}. "
                         f"Known places: {', '.join(known)}.")}

    return {
        "found": True,
        "from": start["name"],
        "to": end["name"],
        "distance_km": calculate_distance(start["lat"], start["lng"],
                                          end["lat"], end["lng"]),
    }


# ---------------------------------------------------------------------------
# Tool 7 - the whole composition, in one call
# ---------------------------------------------------------------------------

def recommend_workforce(skill: str, quantity: int, location: str,
                        on_date: str = "", radius_km: float = 25.0) -> dict:
    """
    Put together a complete workforce for a job, combining crews and
    individual workers.

    Use this when the contractor has told you the trade, how many people
    and where. It returns the actual selection, how many positions are
    filled, and honestly reports a shortfall if there are not enough
    eligible people. Never describe a shortfall as filled.

    Args:
        skill: the trade needed, for example "Mason".
        quantity: how many workers are needed.
        location: the place name, for example "Guntur".
        on_date: the date as YYYY-MM-DD. Leave blank for tomorrow.
        radius_km: how far from the site to search.
    """
    place = _place(location)
    if place is None:
        return _unknown_place(location)

    wanted = _resolve_date(on_date)
    result = compose_workforce(WorkforceRequest(
        skill=skill, quantity=quantity, on_date=wanted,
        location_lat=place["lat"], location_lng=place["lng"],
        location_name=place["name"], max_distance_km=radius_km,
    ))

    # The full candidate lists are useful to the API but too long for the
    # model, and it does not need them to explain the choice.
    result.pop("considered", None)

    return _plain(result)


# ---------------------------------------------------------------------------
# Tools 8-10 - proposing actions that would change something
# ---------------------------------------------------------------------------
#
# These are the only tools that touch anything consequential, and none of
# them actually does it. Each one writes down a proposal and returns its
# id. A person confirms it through the API.
#
# There is deliberately NO tool that confirms a proposal. If the model
# could approve its own proposal, business rule 7 would be a suggestion
# rather than a rule.

def propose_job(contractor_id: str, title: str, skill_required: str,
                workers_required: int, on_date: str, location: str,
                start_time: str = "08:00", wage: float = 0.0,
                site_address: str = "", description: str = "") -> dict:
    """
    Propose creating a job. THIS DOES NOT CREATE THE JOB.

    It writes down what the job would be and returns an action_id. The job
    exists only after a person confirms that action. Tell the contractor
    what you have proposed and ask them to confirm it. Never tell them the
    job has been created.

    Args:
        contractor_id: which contractor the job belongs to, e.g. "CON001".
        title: short description of the work, e.g. "Brickwork, first floor".
        skill_required: the trade needed, e.g. "Mason".
        workers_required: how many people are needed.
        on_date: the date as YYYY-MM-DD.
        location: the place name, e.g. "Guntur".
        start_time: start time as HH:MM. Defaults to 08:00.
        wage: daily wage in rupees, if the contractor gave one.
        site_address: the site address, if known.
        description: any further detail.
    """
    place = _place(location)
    if place is None:
        return _unknown_place(location)

    contractor = fetch_one(
        "select id, company_name from contractors where id = %s",
        (contractor_id,),
    )
    if contractor is None:
        known = fetch_all("select id, company_name from contractors order by id")
        return {"proposed": False,
                "note": (f"No contractor with id {contractor_id}. Known: " +
                         ", ".join(f"{c['id']} ({c['company_name']})"
                                   for c in known))}

    wanted = _resolve_date(on_date)
    payload = {
        "contractor_id": contractor_id, "title": title,
        "description": description, "skill_required": skill_required,
        "workers_required": workers_required,
        "location_name": place["name"], "location_lat": place["lat"],
        "location_lng": place["lng"], "site_address": site_address,
        "date": wanted.isoformat(), "start_time": start_time,
        "wage": wage or None,
    }
    summary = (f"Create a job for {contractor['company_name']}: "
               f"{workers_required} x {skill_required}, {title}, at "
               f"{place['name']} on {wanted.isoformat()} from {start_time}"
               + (f", {wage:.0f} rupees a day" if wage else ""))

    return {"proposed": True,
            **actions.propose("create_job", payload, summary,
                              audit.current_session())}


def propose_offers(job_id: str, worker_ids: list[str] = [],
                   crew_ids: list[str] = []) -> dict:
    """
    Propose sending job offers. THIS DOES NOT SEND ANYTHING.

    It writes down who would be offered the job and returns an action_id.
    Offers go out only after a person confirms that action. Never tell the
    contractor that offers have been sent before that.

    Args:
        job_id: the job to offer, e.g. "J0298".
        worker_ids: individual workers to offer it to, e.g. ["W003"].
        crew_ids: crews to offer it to, e.g. ["RAVI01"].
    """
    job = fetch_one(
        "select id, title, date, skill_required, workers_required "
        "from jobs where id = %s",
        (job_id,),
    )
    if job is None:
        return {"proposed": False, "note": f"There is no job with id {job_id}."}

    if not worker_ids and not crew_ids:
        return {"proposed": False,
                "note": "Give at least one worker_id or crew_id."}

    names = []
    for worker_id in worker_ids:
        row = fetch_one("select name from workers where id=%s", (worker_id,))
        if row is None:
            return {"proposed": False,
                    "note": f"There is no worker with id {worker_id}."}
        names.append(row["name"])
    for crew_id in crew_ids:
        row = fetch_one("select name from crews where id=%s", (crew_id,))
        if row is None:
            return {"proposed": False,
                    "note": f"There is no crew with id {crew_id}."}
        names.append(row["name"])

    payload = {"job_id": job_id, "worker_ids": list(worker_ids),
               "crew_ids": list(crew_ids)}
    summary = (f"Send offers for job {job_id} ({job['title']}, "
               f"{job['date']}) to: " + ", ".join(names))

    return {"proposed": True,
            **actions.propose("send_offers", payload, summary,
                              audit.current_session())}


def check_action_status(action_id: str) -> dict:
    """
    Check whether a proposed action has been confirmed yet.

    Use this before saying anything about what has happened. If the status
    is still "pending", nothing has been done.

    Args:
        action_id: the id returned when the action was proposed.
    """
    found = actions.look_up(action_id)
    if not found.get("found"):
        return found

    return _plain({
        "found": True,
        "action_id": found["id"],
        "action_type": found["action_type"],
        "status": found["status"],
        "summary": found["summary"],
        "result": found.get("result"),
        "error": found.get("error"),
        "note": {
            "pending": "Not done. It is waiting for a person to confirm it.",
            "confirmed": "This was confirmed and carried out.",
            "cancelled": "This was declined. Nothing was changed.",
            "expired": "Nobody answered in time. Nothing was changed.",
            "failed": "Confirmation was attempted but the action failed.",
        }.get(found["status"], ""),
    })


def list_job_offers(job_id: str) -> dict:
    """
    See who has been offered a job and how they answered.

    Args:
        job_id: the job, e.g. "J0298".
    """
    offers = actions.job_offers(job_id)
    return _plain({
        "job_id": job_id,
        "offers": offers,
        "accepted": len([o for o in offers if o["status"] == "accepted"]),
        "confirmed": len([o for o in offers if o["status"] == "confirmed"]),
        "note": ("No offers have been made for this job." if not offers
                 else "status shows where each offer stands."),
    })


# ---------------------------------------------------------------------------
# Tool 11 - independence readiness
# ---------------------------------------------------------------------------

def check_independence_readiness(worker_id: str) -> dict:
    """
    Assess whether a worker has enough verified history to be considered
    for independent work.

    Returns a score out of 100, the five factors behind it, the evidence,
    and a recommendation in words.

    Read the "important" field back to the user. This is a RECOMMENDATION,
    never a change of status: ADAA cannot make anyone independent, the
    worker decides, and their crew membership is unaffected either way.
    Do not say a worker "has been made independent" or "is now
    independent" -- nothing here does that.

    The score is a prototype decision-support figure. It has not been
    scientifically validated, and you should say so if you quote it.

    Args:
        worker_id: the worker's ADAA id, for example "W001".
    """
    return _plain(independence.assess(worker_id, save=True))


# Everything Gemini is allowed to call.
#
# Each one is wrapped so the call is written to agent_actions: which tool,
# what arguments, what came back, how long it took, and whether it worked.
# The wrapper keeps the name, docstring and signature, which is what the
# SDK reads to describe the tool to Gemini.
ALL_TOOLS = [
    logged(search_workers),
    logged(search_crews),
    logged(get_worker_profile),
    logged(get_crew_profile),
    logged(check_availability),
    logged(distance_between),
    logged(recommend_workforce),
    # Proposing only. None of these changes anything by itself.
    logged(propose_job),
    logged(propose_offers),
    logged(check_action_status),
    logged(list_job_offers),
    logged(check_independence_readiness),
]
