"""Cloud backend: accounts, device keys, ingest from the car, consent, data quality, export."""
import numpy as np
import pytest

pytest.importorskip("sqlalchemy")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DMS_CLOUD_DIR", str(tmp_path))
    from dms.cloud.app import create_app

    url = f"sqlite:///{tmp_path / 'cloud.db'}"
    c = TestClient(create_app(url))
    c.db_url = url
    return c


def _user(c, email="driver@example.com"):
    r = c.post("/api/auth/register", json={"email": email, "password": "Sleepy-Driver-42", "name": "Test"})
    assert r.status_code == 200, r.text
    return {"Authorization": "Bearer " + r.json()["token"]}


def _device(c, h):
    r = c.post("/api/devices", json={"name": "dash cam", "kind": "dashcam"}, headers=h)
    return {"Authorization": "Bearer " + r.json()["device_key"]}


def _window(label=2, source="kss", face=1.0, frontal=0.9, seconds=30, switches=0, **meta):
    n = int(seconds * 10)
    raw = [[0.27, 0.05, 1.0, 2.0, 0.0, 0.5, 0.5, 0.1, 0.0] for _ in range(n)]
    present = [i < face * n for i in range(n)]
    return {"id": f"s{label}{source}{face}{frontal}{seconds}{switches}{len(meta)}", "session": "trip1", "label": label, "label_source": source,
            "raw": raw, "present": present, "fps": 10.0, "profile": [0.28, 0.05, 0.0, 0.0, 120],
            "meta": {"frontal_frac": frontal, "max_faces": 1, "driver_switches": switches, **meta}}


def test_register_login_and_bad_passwords(client):
    assert client.post("/api/auth/register", json={"email": "a@b.co", "password": "short"}).status_code == 400
    _user(client)
    assert client.post("/api/auth/register", json={"email": "driver@example.com", "password": "Sleepy-Driver-42"}).status_code == 409
    assert client.post("/api/auth/login", json={"email": "driver@example.com", "password": "Sleepy-Driver-42"}).status_code == 200
    for _ in range(5):
        assert client.post("/api/auth/login", json={"email": "driver@example.com", "password": "wrong"}).status_code == 401
    # brute force blocked, even with the right password
    assert client.post("/api/auth/login", json={"email": "driver@example.com", "password": "Sleepy-Driver-42"}).status_code == 429
    assert client.get("/api/me").status_code == 401
    assert client.get("/api/me", headers={"Authorization": "Bearer forged.9999999999.abc"}).status_code == 401


def test_ingest_trips_events_and_idempotency(client):
    h = _user(client)
    dev = _device(client, h)
    body = {"events": [{"id": "e1", "session": "trip1", "t_wall": 1000.0, "t_session": 12.0, "kind": "drowsy", "severity": "warning",
                        "payload": {"detail": "drowsiness 70%"}}],
            "trips": [{"session": "trip1", "t_start": 900.0, "t_end": 1100.0, "summary": {"safety_score": 81, "alerts_total": 1}, "kss": 6}]}
    assert client.post("/api/ingest", json=body, headers=dev).json()["events"] == 1
    assert client.post("/api/ingest", json=body, headers=dev).json()["events"] == 0  # resend after a lost reply: no duplicate
    assert client.post("/api/ingest", json=body, headers={"Authorization": "Bearer dmsk_wrong"}).status_code == 401
    trips = client.get("/api/trips", headers=h).json()
    assert trips[0]["safety_score"] == 81 and trips[0]["kss"] == 6
    t = client.get("/api/trips/trip1", headers=h).json()
    assert t["events"][0]["kind"] == "drowsy"
    # the driver rated the alert in the car: the rating is attached to that alert, not stored as an alert itself
    fb = {"events": [{"id": "f1", "session": "trip1", "t_wall": 1001.0, "t_session": 12.0, "kind": "feedback", "severity": "info",
                      "payload": {"alert": "drowsy", "verdict": "correct"}}]}
    client.post("/api/ingest", json=fb, headers=dev)
    t = client.get("/api/trips/trip1", headers=h).json()
    assert [e["kind"] for e in t["events"]] == ["drowsy"] and t["events"][0]["feedback"] == "correct"
    assert client.post("/api/events/e1/feedback", json={"feedback": "false_alarm"}, headers=h).status_code == 200
    assert client.get("/api/stats", headers=h).json()["feedback"] == {"false_alarm": 1}


def test_accounts_are_isolated(client):
    h1, h2 = _user(client, "one@example.com"), _user(client, "two@example.com")
    dev1 = _device(client, h1)
    client.post("/api/ingest", json={"trips": [{"session": "t-one", "summary": {}}]}, headers=dev1)
    assert client.get("/api/trips/t-one", headers=h2).status_code == 404
    dev2 = _device(client, h2)  # a device of account 2 can't overwrite account 1's trip
    client.post("/api/ingest", json={"trips": [{"session": "t-one", "summary": {"safety_score": 0}}]}, headers=dev2)
    assert client.get("/api/trips/t-one", headers=h1).json()["summary"] == {}
    devs = client.get("/api/devices", headers=h1).json()
    assert client.delete(f"/api/devices/{devs[0]['id']}", headers=h2).status_code == 404


def test_training_samples_need_consent_and_pass_quality(client):
    h = _user(client)
    dev = _device(client, h)
    good = _window()
    r = client.post("/api/ingest", json={"samples": [good]}, headers=dev).json()
    assert r["samples"] == {"accepted": 0, "rejected": 0, "skipped_no_consent": 1}  # no consent: dropped
    client.patch("/api/me", json={"consent_training": True}, headers=h)
    bad = [_window(face=0.5), _window(frontal=0.3), _window(seconds=10), _window(switches=1), _window(speed_kmh=0.0), _window(label=7)]
    r = client.post("/api/ingest", json={"samples": [good] + bad}, headers=dev).json()
    assert r["samples"]["accepted"] == 1 and r["samples"]["rejected"] == len(bad)
    _user(client, "boss@example.com")
    from dms.cloud.app import make_admin

    make_admin("boss@example.com", client.db_url)
    ha = {"Authorization": "Bearer " + client.post("/api/auth/login", json={"email": "boss@example.com", "password": "Sleepy-Driver-42"}).json()["token"]}
    ds = client.get("/api/admin/dataset", headers=ha).json()
    assert ds["accepted_by_label"] == {"drowsy": 1}
    assert any("not moving" in k for k in ds["rejected_by_reason"]) and any("changed" in k for k in ds["rejected_by_reason"])
    assert client.get("/api/admin/dataset", headers=h).status_code == 403
    # withdrawing consent deletes what was shared
    client.patch("/api/me", json={"consent_training": False}, headers=h)
    assert client.get("/api/stats", headers=h).json()["training_samples_given"] == 0


def test_export_matches_training_format(client, tmp_path):
    h = _user(client)
    client.patch("/api/me", json={"consent_training": True}, headers=h)
    dev = _device(client, h)
    client.post("/api/ingest", json={"samples": [_window(label=0, source="false_alarm"), _window(label=2, source="kss")]}, headers=dev)
    from dms.cloud.export_dataset import export

    c = export(tmp_path / "fleet", client.db_url, test_frac=0.0)
    assert sum(c.values()) == 2
    from dms.training.train_temporal import load_dir

    vids = load_dir(tmp_path / "fleet" / "train", binary=True)
    assert sorted(v["y"] for v in vids) == [0, 1] and vids[0]["x"].shape == (300, 9)
    assert vids[0]["subject"].startswith("drv_")  # pseudonymous, no email in the export


def test_edge_store_queues_samples_for_sync(tmp_path):
    from dms.storage import EventStore

    st = EventStore(tmp_path / "e.db")
    st.save_trip("trip1", {"safety_score": 90})
    st.set_trip_kss("trip1", 7)
    sid = st.add_sample("trip1", {"label": 2, "label_source": "kss"})
    assert st.unsynced_samples()[0]["id"] == sid
    ev, trips = st.unsynced()
    assert trips[0]["summary"]["kss"] == 7
    st.mark_samples_synced([sid])
    assert st.unsynced_samples() == []


def test_training_window_snapshot():
    from dms.features.geometry import FrameFeatures
    from dms.features.temporal import DriverProfile
    from dms.pipeline import TrainingWindow

    w = TrainingWindow(30, 10)
    f = FrameFeatures(face=False)
    for i in range(400):
        w.update(i / 10, f, False, None, 640)
    snap = w.snapshot(DriverProfile(n_samples=50))
    assert len(snap["raw"]) == 300 and snap["raw"][0][0] is None and snap["meta"]["frontal_frac"] == 0.0
    from dms.cloud.quality import check

    ok, why, _ = check({**snap, "label": 0, "label_source": "kss"})
    assert not ok and "face" in why  # no face in view -> not a training sample
