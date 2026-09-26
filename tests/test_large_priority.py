"""Large (arbitrary-precision) rule priorities.

The API accepts any non-negative integer priority; values beyond the signed-64
SQLite INTEGER range (e.g. 2**63) must persist verbatim, round-trip as exact
integers, and keep adjudicating same-CIDR ties by (priority, submission order).
Legacy databases whose ``rules.priority`` is INTEGER are upgraded in place at
startup.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile

import pytest
from fastapi.testclient import TestClient

from app import db

BIG = 9223372036854775808  # 2**63, one above the SQLite INTEGER maximum
HUGE = 2**128  # far beyond any fixed-width integer type


@pytest.fixture()
def client() -> TestClient:
    fd, path = tempfile.mkstemp(suffix=".db", prefix="ppa-large-prio-")
    os.close(fd)
    os.remove(path)  # let sqlite create it
    db.reset_db(path)
    from app.main import app

    with TestClient(app) as c:
        yield c
    for suffix in ("", "-wal", "-shm"):
        try:
            os.remove(path + suffix)
        except OSError:
            pass


def _rules() -> list[dict]:
    return [
        {"rule_id": "low", "cidr": "10.0.0.0/8", "action": "deny", "priority": BIG},
        {"rule_id": "win", "cidr": "10.0.0.0/8", "action": "permit", "priority": 5},
        {"rule_id": "huge", "cidr": "10.1.0.0/16", "action": "permit", "priority": HUGE},
    ]


def test_large_priority_is_saved_and_returned_as_exact_integer(client: TestClient) -> None:
    r = client.post("/api/analyze", json={"name": "big", "rules": _rules()})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["persisted"] is True
    by_id = {rule["rule_id"]: rule for rule in data["rules"]}
    assert by_id["low"]["priority"] == BIG
    assert by_id["win"]["priority"] == 5
    assert by_id["huge"]["priority"] == HUGE

    aid = data["id"]

    # GET full version: exact integers in both parsed JSON and raw text.
    fetched = client.get(f"/api/analyses/{aid}")
    assert fetched.status_code == 200
    fetched_rules = {x["rule_id"]: x for x in fetched.json()["rules"]}
    assert fetched_rules["low"]["priority"] == BIG
    assert fetched_rules["huge"]["priority"] == HUGE
    assert isinstance(fetched_rules["low"]["priority"], int)
    assert str(BIG) in fetched.text
    assert str(HUGE) in fetched.text

    # GET filtered rule rows: parsed back from the TEXT column to ints.
    listed = client.get(f"/api/analyses/{aid}/rules")
    assert listed.status_code == 200
    listed_rules = {x["rule_id"]: x for x in listed.json()["rules"]}
    assert listed_rules["low"]["priority"] == BIG
    assert listed_rules["huge"]["priority"] == HUGE
    assert isinstance(listed_rules["huge"]["priority"], int)
    assert str(HUGE) in listed.text


def test_priority_column_is_text_in_storage(client: TestClient) -> None:
    r = client.post("/api/analyze", json={"rules": _rules()})
    aid = r.json()["id"]
    with db._connect() as conn:
        col_type = [
            row["type"]
            for row in conn.execute("PRAGMA table_info(rules)")
            if row["name"] == "priority"
        ][0]
        assert col_type.upper() == "TEXT"
        # Stored verbatim as the canonical decimal representation.
        values = dict(
            conn.execute("SELECT rule_id, priority FROM rules WHERE analysis_id = ?", (aid,))
        )
        assert values["low"] == str(BIG)
        assert values["huge"] == str(HUGE)
        assert values["win"] == "5"


def test_duplicate_submission_reuses_the_same_version(client: TestClient) -> None:
    first = client.post("/api/analyze", json={"name": "first", "rules": _rules()})
    assert first.status_code == 200, first.text
    first_data = first.json()

    # Same ruleset, different name and (verdict-neutrally) reordered: the
    # fingerprint collapses it onto the same stored version.
    second = client.post(
        "/api/analyze",
        json={"name": "second-name-ignored", "rules": list(reversed(_rules()))},
    )
    assert second.status_code == 200, second.text
    second_data = second.json()
    assert second_data["id"] == first_data["id"]
    assert second_data["name"] == "first"
    assert second_data["created_at"] == first_data["created_at"]
    assert [r["rule_id"] for r in second_data["rules"]] == [
        r["rule_id"] for r in first_data["rules"]
    ]
    assert client.get("/api/analyses").json()["total"] == 1


def test_same_cidr_adjudicated_by_priority_then_order_with_large_values(
    client: TestClient,
) -> None:
    # The big-number rule loses the same-CIDR tie to the small-number rule,
    # even though it was submitted first.
    r = client.post(
        "/api/analyze",
        json={
            "rules": [
                {"rule_id": "bigdeny", "cidr": "10.0.0.0/8", "action": "deny", "priority": BIG},
                {"rule_id": "smallpermit", "cidr": "10.0.0.0/8", "action": "permit", "priority": 5},
            ]
        },
    )
    assert r.status_code == 200, r.text
    by_id = {x["rule_id"]: x for x in r.json()["rules"]}
    assert by_id["smallpermit"]["role"] == "winner"
    assert by_id["smallpermit"]["status"] == "active"
    assert by_id["bigdeny"]["role"] == "duplicate"
    assert by_id["bigdeny"]["status"] == "shadowed"

    # Equal oversized priorities: original submission order breaks the tie.
    first_order = client.post(
        "/api/analyze",
        json={
            "rules": [
                {"rule_id": "a", "cidr": "172.16.0.0/12", "action": "permit", "priority": HUGE},
                {"rule_id": "b", "cidr": "172.16.0.0/12", "action": "deny", "priority": HUGE},
            ]
        },
    ).json()
    winners = {x["rule_id"]: x for x in first_order["rules"] if x["role"] == "winner"}
    assert set(winners) == {"a"}

    swapped = client.post(
        "/api/analyze",
        json={
            "rules": [
                {"rule_id": "b", "cidr": "172.16.0.0/12", "action": "deny", "priority": HUGE},
                {"rule_id": "a", "cidr": "172.16.0.0/12", "action": "permit", "priority": HUGE},
            ]
        },
    ).json()
    winners = {x["rule_id"]: x for x in swapped["rules"] if x["role"] == "winner"}
    assert set(winners) == {"b"}
    # The submission-order swap distinguishes the fingerprint, as for normal ints.
    assert swapped["ruleset_hash"] != first_order["ruleset_hash"]


def test_save_false_matches_persisted_result(client: TestClient) -> None:
    saved = client.post("/api/analyze", json={"rules": _rules()}).json()
    draft = client.post("/api/analyze", json={"rules": _rules(), "save": False}).json()
    assert draft["persisted"] is False
    for field in ("ruleset_hash", "analyzer_version", "summary", "findings",
                  "minimal_conflict_pairs"):
        assert draft[field] == saved[field]
    assert [
        {k: r[k] for k in ("rule_id", "priority", "role", "status")} for r in draft["rules"]
    ] == [
        {k: r[k] for k in ("rule_id", "priority", "role", "status")} for r in saved["rules"]
    ]
    # Nothing was written by the dry run beyond the one saved version.
    assert client.get("/api/analyses").json()["total"] == 1


def test_negative_priority_is_rejected(client: TestClient) -> None:
    r = client.post(
        "/api/analyze",
        json={"rules": [{"cidr": "10.0.0.0/8", "action": "permit", "priority": -1}]},
    )
    assert r.status_code == 422


def test_legacy_integer_priority_database_is_upgraded() -> None:
    """A pre-upgrade database (priority INTEGER) migrates and keeps its rows."""
    fd, path = tempfile.mkstemp(suffix=".db", prefix="ppa-legacy-prio-")
    os.close(fd)
    os.remove(path)
    try:
        conn = sqlite3.connect(path)
        conn.executescript(
            """
CREATE TABLE analyses (
    id TEXT PRIMARY KEY, name TEXT, ruleset_hash TEXT NOT NULL,
    analyzer_version TEXT NOT NULL, created_at TEXT NOT NULL,
    rule_count INTEGER NOT NULL, result_json TEXT NOT NULL
);
CREATE TABLE rules (
    analysis_id TEXT NOT NULL, rule_index INTEGER NOT NULL, rule_id TEXT NOT NULL,
    cidr TEXT NOT NULL, ip_version INTEGER NOT NULL, action TEXT NOT NULL,
    priority INTEGER NOT NULL, source_tag TEXT NOT NULL, role TEXT NOT NULL,
    status TEXT NOT NULL, covered TEXT NOT NULL, has_conflict INTEGER NOT NULL,
    PRIMARY KEY (analysis_id, rule_index)
);
CREATE TABLE findings (
    analysis_id TEXT NOT NULL, seq INTEGER NOT NULL, kind TEXT NOT NULL,
    severity TEXT NOT NULL, rule_id TEXT, other_rule_id TEXT,
    behavioral INTEGER NOT NULL, payload_json TEXT NOT NULL,
    PRIMARY KEY (analysis_id, seq)
);
CREATE INDEX idx_rules_analysis ON rules(analysis_id);
CREATE INDEX idx_findings_analysis ON findings(analysis_id);
"""
        )
        conn.execute(
            "INSERT INTO analyses VALUES (?,?,?,?,?,?,?)",
            ("an_legacyrow0000000", "legacy", "legacyhash", "1.0.0",
             "2026-01-01T00:00:00Z", 1, "{}"),
        )
        conn.execute(
            "INSERT INTO rules VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            ("an_legacyrow0000000", 0, "old", "10.0.0.0/8", 4, "permit",
             100, "default", "winner", "active", "16777216", 0),
        )
        conn.commit()
        conn.close()

        # Startup initialisation performs the in-place upgrade.
        db.set_db_path(path)
        db.init_db()

        with db._connect() as upgraded:
            col_type = [
                row["type"]
                for row in upgraded.execute("PRAGMA table_info(rules)")
                if row["name"] == "priority"
            ][0]
            assert col_type.upper() == "TEXT"
            legacy = upgraded.execute(
                "SELECT rule_id, priority FROM rules"
            ).fetchone()
            assert legacy["rule_id"] == "old"
            assert legacy["priority"] == "100"
            # The supporting index survived the table rebuild.
            indexes = {
                row["name"]
                for row in upgraded.execute("PRAGMA index_list('rules')")
            }
            assert "idx_rules_analysis" in indexes

        # The migrated database accepts oversized priorities afterwards.
        from fastapi.testclient import TestClient as _TC

        from app.main import app

        with _TC(app) as c:
            r = c.post(
                "/api/analyze",
                json={"rules": [{"cidr": "192.0.2.0/24", "action": "deny", "priority": BIG}]},
            )
            assert r.status_code == 200, r.text
            aid = r.json()["id"]
            row = c.get(f"/api/analyses/{aid}/rules").json()["rules"][0]
            assert row["priority"] == BIG

        # Migration is idempotent across restarts.
        db.init_db()
        with db._connect() as again:
            assert again.execute("SELECT COUNT(*) AS c FROM rules").fetchone()["c"] == 2
    finally:
        reset_fd, reset_path = tempfile.mkstemp(suffix=".db")
        os.close(reset_fd)
        db.set_db_path(reset_path)  # avoid later writes touching the removed temp path
        for suffix in ("", "-wal", "-shm"):
            try:
                os.remove(path + suffix)
            except OSError:
                pass
