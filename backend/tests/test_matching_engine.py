"""
Tests for the matching engine running against the real database.

These are the tests that matter most for the project, because they check
the promises ADAA makes to a contractor: that everyone recommended is
genuinely eligible, that nobody is counted twice, and that a shortfall is
admitted rather than hidden.

Skipped automatically if the database cannot be reached.
"""

from datetime import date, timedelta

import pytest

from app.agent.matching import WorkforceRequest, compose_workforce, find_crews, find_workers
from app.database import fetch_all, fetch_one

GUNTUR_LAT, GUNTUR_LNG = 16.3067, 80.4365


@pytest.fixture(scope="module", autouse=True)
def require_database():
    try:
        fetch_one("select 1 as ok")
    except Exception:
        pytest.skip("database not reachable")


def tomorrow() -> date:
    return date.today() + timedelta(days=1)


def mason_request(quantity=8, radius=25.0) -> WorkforceRequest:
    return WorkforceRequest(
        skill="Mason", quantity=quantity, on_date=tomorrow(),
        location_lat=GUNTUR_LAT, location_lng=GUNTUR_LNG,
        location_name="Guntur", max_distance_km=radius,
    )


# --- the filters -----------------------------------------------------------

def test_every_returned_worker_is_verified_available_and_skilled():
    """
    Business rules 1 and 2: never claim an unavailable worker is free, and
    never use an unverified skill. We check each returned worker against
    the database directly.
    """
    workers = find_workers(mason_request())
    assert workers, "expected some masons"

    for candidate in workers:
        row = fetch_one(
            """
            select w.verification_status,
                   (select count(*) from worker_skills ws
                      join skills s on s.id = ws.skill_id
                     where ws.worker_id = w.id
                       and lower(s.name) = 'mason'
                       and ws.verification_status = 'verified') as verified_mason,
                   (select count(*) from availability a
                     where a.worker_id = w.id and a.date = %s
                       and a.status = 'available') as free_that_day
              from workers w where w.id = %s
            """,
            (tomorrow(), candidate.id),
        )
        assert row["verification_status"] == "verified", candidate.id
        assert row["verified_mason"] == 1, candidate.id
        assert row["free_that_day"] == 1, candidate.id


def test_nobody_is_returned_from_beyond_the_search_radius():
    for candidate in find_workers(mason_request(radius=25.0)):
        assert candidate.distance_km <= 25.0


def test_a_tiny_radius_returns_fewer_people_than_a_large_one():
    wide = find_workers(mason_request(radius=60.0))
    narrow = find_workers(mason_request(radius=3.0))

    assert len(narrow) <= len(wide)


def test_an_unknown_skill_returns_nothing_rather_than_guessing():
    request = mason_request()
    request.skill = "Astronaut"

    assert find_workers(request) == []
    assert find_crews(request) == []


# --- crews -----------------------------------------------------------------

def test_ravi_crew_supplies_exactly_six_masons():
    """Specification section 15."""
    crews = {c.id: c for c in find_crews(mason_request())}

    assert "RAVI01" in crews
    assert crews["RAVI01"].supply == 6


def test_crew_supply_counts_only_qualified_available_members():
    """
    A crew's supply must come from its members' own verified skills, not
    from the crew's headcount or its reputation (business rules 2 and 3).
    """
    for crew in find_crews(mason_request()):
        assert crew.supply == len(crew.evidence["member_ids"])
        assert crew.supply <= fetch_one(
            "select count(*) as n from crew_members where crew_id=%s and status='active'",
            (crew.id,),
        )["n"]


def test_workers_carry_their_crew_and_its_leader():
    """
    A contractor dealing with an individual worker still deals with that
    worker's crew leader, so every suggested worker must say which crew
    they belong to and who leads it -- from the database, never guessed.
    """
    workers = find_workers(mason_request())
    assert workers, "expected some masons"

    for candidate in workers:
        evidence = candidate.evidence
        assert "crew_name" in evidence
        assert "crew_leader" in evidence
        assert "crew_leader_available" in evidence

        membership = fetch_one(
            """
            select c.id as crew_id, c.name as crew_name,
                   leader.name as leader_name
              from crew_members cm
              join crews c on c.id = cm.crew_id
              left join workers leader on leader.id = c.leader_worker_id
             where cm.worker_id = %s and cm.status = 'active'
             order by cm.joined_at desc, c.id
             limit 1
            """,
            (candidate.id,),
        )
        if membership is None:
            assert evidence["crew_name"] is None, candidate.id
            assert evidence["crew_leader"] is None, candidate.id
            assert evidence["crew_leader_available"] is None, candidate.id
        else:
            assert evidence["crew_id"] == membership["crew_id"], candidate.id
            assert evidence["crew_name"] == membership["crew_name"], candidate.id
            assert evidence["crew_leader"] == membership["leader_name"], candidate.id


def test_crew_leader_availability_comes_from_the_availability_table():
    """
    Business rule 1: whether a crew's leader is free comes from the
    availability table, nowhere else.
    """
    for candidate in find_workers(mason_request()):
        if not candidate.evidence["crew_leader"]:
            continue

        leader = fetch_one(
            """
            select (select a.status from availability a
                     where a.worker_id = c.leader_worker_id
                       and a.date = %s) as status
              from crews c where c.id = %s
            """,
            (tomorrow(), candidate.evidence["crew_id"]),
        )
        assert candidate.evidence["crew_leader_available"] == \
            (leader["status"] == "available"), candidate.id


def test_crew_results_name_their_leader():
    crews = find_crews(mason_request())
    assert crews, "expected some crews"

    for crew in crews:
        assert "crew_leader" in crew.evidence
        assert "crew_leader_available" in crew.evidence

        row = fetch_one(
            """
            select leader.name as leader_name
              from crews c
              left join workers leader on leader.id = c.leader_worker_id
             where c.id = %s
            """,
            (crew.id,),
        )
        assert crew.evidence["crew_leader"] == row["leader_name"], crew.id


# --- composition -----------------------------------------------------------

def test_the_eight_mason_scenario_is_filled():
    """The first demonstration scenario, section 14."""
    result = compose_workforce(mason_request(quantity=8))

    assert result["filled"] == 8
    assert result["shortfall"] == 0
    assert result["complete"] is True


def test_the_eight_mason_scenario_uses_ravi_crew_plus_individuals():
    """
    Section 15 says the remaining positions are filled by individual
    workers, not by breaking up a second crew.
    """
    result = compose_workforce(mason_request(quantity=8))
    picked = result["selection"]

    crews = [c for c in picked if c["kind"] == "crew"]
    individuals = [c for c in picked if c["kind"] == "worker"]

    assert [c["id"] for c in crews] == ["RAVI01"]
    assert crews[0]["supply"] == 6
    assert len(individuals) == 2


def test_nobody_is_counted_twice():
    """
    A crew member must never also appear as an individual. Counting one
    person twice would overstate the workforce, which rule 1 forbids.
    """
    result = compose_workforce(mason_request(quantity=8))

    individual_ids = {c["id"] for c in result["selection"] if c["kind"] == "worker"}
    crew_member_ids = set()
    for entry in result["selection"]:
        if entry["kind"] == "crew":
            crew_member_ids.update(entry["evidence"]["member_ids"])

    assert individual_ids.isdisjoint(crew_member_ids)


def test_the_supplied_numbers_add_up_to_the_reported_total():
    result = compose_workforce(mason_request(quantity=8))

    assert sum(c["supply"] for c in result["selection"]) == result["filled"]


def test_a_crew_never_contributes_more_than_it_has():
    result = compose_workforce(mason_request(quantity=8))

    for entry in result["selection"]:
        if entry["kind"] == "crew":
            assert entry["supply"] <= entry["evidence"]["qualified_available_members"]


def test_an_impossible_request_reports_a_shortfall_instead_of_inventing_people():
    """
    Business rule 1 and rule 9. Asking for 500 masons must produce an
    honest shortfall, never a padded list.
    """
    result = compose_workforce(mason_request(quantity=500))

    assert result["complete"] is False
    assert result["shortfall"] > 0
    assert result["filled"] + result["shortfall"] == 500
    assert result["filled"] == sum(c["supply"] for c in result["selection"])


def test_asking_for_one_worker_returns_one_worker():
    result = compose_workforce(mason_request(quantity=1))

    assert result["filled"] == 1
    assert len(result["selection"]) == 1


def test_results_are_ordered_best_first():
    result = compose_workforce(mason_request(quantity=500))
    workers = [c for c in result["selection"] if c["kind"] == "worker"]
    scores = [w["match_score"] for w in workers]

    assert scores == sorted(scores, reverse=True)


def test_the_same_request_twice_gives_the_same_answer():
    """
    Specification section 23: the agent must be consistent when the data
    has not changed. If this fails, the evaluation is meaningless.
    """
    first = compose_workforce(mason_request(quantity=8))
    second = compose_workforce(mason_request(quantity=8))

    assert first["selection"] == second["selection"]


def test_every_selected_candidate_carries_its_evidence():
    """
    The agent has to explain its recommendation using real data, so every
    entry must arrive with the numbers behind it.
    """
    result = compose_workforce(mason_request(quantity=8))

    for entry in result["selection"]:
        assert entry["evidence"], entry["id"]
        assert 0 <= entry["match_score"] <= 100
        assert set(entry["scores"]) == {
            "skill", "availability", "reliability", "rating",
            "proximity", "experience",
        }
