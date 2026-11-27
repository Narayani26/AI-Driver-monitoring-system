import json
import numpy as np
import pandas as pd
import pytest
import torch

from dms.features.geometry import LEFT_EYE, RIGHT_EYE, eye_aspect_ratio, rotation_to_euler, compute_features
from dms.features.temporal import Calibrator, DriverProfile, EyeClosureTracker, FeatureSequence, GlanceTracker, YawnTracker
from dms.fusion.risk import RiskFusion, RiskInputs
from dms.models import checkpoint
from dms.models.networks import DrowsinessGRU, DrowsinessTransformer, FaceStateNet
from dms.data.datasets import subject_split, assert_no_leakage

PROF = DriverProfile(ear_open=0.30, mar_rest=0.05)


def _eye(lm, idx, w, h):
    cx, cy = 100.0 + idx[0], 100.0
    p = [(cx - w / 2, cy), (cx - w / 6, cy - h / 2), (cx + w / 6, cy - h / 2), (cx + w / 2, cy), (cx + w / 6, cy + h / 2), (cx - w / 6, cy + h / 2)]
    for i, xy in zip(idx, p):
        lm[i, :2] = xy


def test_ear_matches_definition():
    lm = np.zeros((478, 3), np.float32)
    _eye(lm, LEFT_EYE, 30, 9)
    assert eye_aspect_ratio(lm, LEFT_EYE) == pytest.approx(9 / 30, rel=1e-5)


def test_euler_identity_and_yaw():
    assert rotation_to_euler(np.eye(3)) == pytest.approx((0, 0, 0), abs=1e-6)
    a = np.radians(30)
    Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    assert rotation_to_euler(Ry)[1] == pytest.approx(30, abs=1e-4)


def test_compute_features_without_iris_or_pose():
    lm = np.random.rand(468, 3).astype(np.float32) * 100
    f = compute_features(lm)
    assert f.face and np.isnan(f.gaze_h) and np.isnan(f.yaw)
    assert f.vector().shape == (9,)


def _run_eyes(signal_fn, seconds, fps=30):
    tr = EyeClosureTracker()
    for i in range(int(seconds * fps)):
        t = i / fps
        tr.update(t, signal_fn(t), PROF)
    return tr


def test_blinks_counted_and_perclos():
    # 0.15 s blink every 4 s -> 15 blinks/min, PERCLOS ~ 3.75 %
    tr = _run_eyes(lambda t: 0.05 if (t % 4.0) < 0.15 else 0.30, 60)
    assert 13 <= tr.blink_rate() <= 16
    assert 0.02 < tr.perclos() < 0.06
    assert tr.microsleep_events == 0


def test_microsleep_detected():
    tr = _run_eyes(lambda t: 0.05 if 10 < t < 12 else 0.30, 20)
    assert tr.microsleep_events == 1


def test_calibrator():
    c = Calibrator(duration=2.0, min_samples=10)
    from dms.features.geometry import FrameFeatures

    for i in range(90):
        c.update(i / 30, FrameFeatures(face=True, ear=0.25, mar=0.04, yaw=5.0, pitch=-3.0))
    assert c.done and c.profile.ear_open == pytest.approx(0.25) and c.profile.yaw0 == pytest.approx(5.0)


def test_yawn_needs_duration():
    y = YawnTracker()
    for i in range(30):  # 1 s open -> talking, not a yawn
        y.update(i / 30, 0.8, float("nan"))
    y.update(1.1, 0.1, float("nan"))
    assert y.count() == 0
    for i in range(90):  # 3 s open
        y.update(2 + i / 30, 0.8, float("nan"))
    assert y.count() == 1


def test_glance_long_and_vats():
    g = GlanceTracker()
    st = None
    for i in range(int(4 * 30)):  # 4 s looking 40 deg to the side
        st = g.update(i / 30, 40.0, 0.0, PROF)
    assert st.long_distraction
    g = GlanceTracker()
    for i in range(int(30 * 30)):  # 2 s away every 4 s -> 15 s off-road in 30 s, never >= 3 s
        t = i / 30
        st = g.update(t, 40.0 if (t % 4) < 2 else 0.0, 0.0, PROF)
    assert st.vats_distraction and not st.long_distraction


def test_feature_sequence_resamples():
    s = FeatureSequence(length=50, rate_hz=10)
    for i in range(300):  # 10 s @ 30 FPS -> 100 samples, buffer keeps 50
        s.update(i / 30, np.ones(9, np.float32) * 0.3, PROF)
    assert s.full and s.array().shape == (50, 9)
    assert s.array()[0, 0] == pytest.approx(1.0)  # ear normalised by ear_open


def test_risk_levels_and_hysteresis():
    rf = RiskFusion()
    out = rf.update(0.0, RiskInputs())
    assert out.level == "OK"
    out = rf.update(0.1, RiskInputs(microsleep=True, closed_duration=1.2))
    assert out.level == "CRITICAL" and out.reasons[0].startswith("MICROSLEEP")
    for i in range(20):  # 10 s later: still elevated (recent microsleep), not reset
        out = rf.update(0.2 + i * 0.5, RiskInputs())
    assert out.level == "WARNING" and any("recent microsleep" in r for r in out.reasons)
    for i in range(240):  # 2 min later the floor has decayed and hysteresis lets it return to OK
        out = rf.update(10.0 + i * 0.5, RiskInputs())
    assert out.level == "OK"


def test_perclos_not_inflated_early():
    # one 2.5 s closure after 10 s of driving must not look like 25 % PERCLOS
    tr = _run_eyes(lambda t: 0.05 if 8 < t < 10.5 else 0.30, 11)
    assert tr.perclos() < 0.1


def test_alert_cooldown_and_debounce():
    from dms.fusion.alerts import AlertManager
    from dms.fusion.risk import RiskOutput

    am, risk = AlertManager(), RiskOutput(0, "OK", 0, 0, 0)
    kw = dict(risk=risk, microsleep=False, closed_duration=0, long_distraction=False, vats=False, off_road_duration=0,
              yawns_5min=0, face=True, alertness=None, emotion_probs_affect=0.0)
    fired = [a.kind for i in range(40) for a in am.update(i * 0.1, **kw, behaviours={"phone": 0.9})]
    assert fired == ["phone"]  # debounced 1 s, then only once inside the cooldown
    fired = [a.kind for a in am.update(5.0, **{**kw, "microsleep": True}, behaviours={})]
    assert fired == ["microsleep"]


def test_phone_showing_face_is_ignored():
    from dms.perception.objects import Detection, behaviour_scores

    face = (100, 100, 200, 200)
    screen = Detection("cell phone", "phone", 0.9, (50, 50, 260, 300))  # phone screen displaying the face
    held = Detection("cell phone", "phone", 0.8, (210, 150, 260, 230))  # phone next to the ear
    assert behaviour_scores([screen], face, (480, 640)) == {}
    assert behaviour_scores([held], face, (480, 640)) == {"phone": 0.8}


def test_multitask_masked_loss():
    m = FaceStateNet("mobilenetv3_small_100", pretrained=False)
    x = torch.randn(4, 3, 96, 96)
    out = m(x)
    tgt = {"eye": torch.tensor([0, 1, -1, -1]), "yawn": torch.tensor([-1, -1, 1, -1]), "emotion": torch.tensor([-1, -1, -1, -1])}
    loss, parts = m.loss(out, tgt)
    assert torch.isfinite(loss) and set(parts) == {"eye", "yawn"}
    loss.backward()
    assert m.heads["emotion"][1].weight.grad is None or m.heads["emotion"][1].weight.grad.abs().sum() == 0


@pytest.mark.parametrize("cls", [DrowsinessTransformer, DrowsinessGRU])
def test_temporal_models(cls):
    m = cls(in_dim=9)
    assert m(torch.randn(2, 300, 9)).shape == (2, 3)


def test_checkpoint_roundtrip(tmp_path):
    m = DrowsinessTransformer(in_dim=9).eval()
    checkpoint.save(tmp_path / "t.pt", m, "temporal", {"kind": "transformer", "in_dim": 9})
    m2, ck = checkpoint.load(tmp_path / "t.pt")
    x = torch.randn(1, 50, 9)
    assert torch.allclose(m(x), m2(x), atol=1e-6)


def test_subject_split_no_leakage():
    df = pd.DataFrame({"path": [f"{i}.png" for i in range(400)], "subject": [f"s{i % 40}" for i in range(400)], "source": "x", "split": "", "eye": 1, "yawn": -1, "emotion": -1, "cabin": -1})
    out = subject_split(df, seed=1)
    assert_no_leakage(out)
    assert set(out.split) == {"train", "val", "test"}


def test_random_split_is_leaky_and_subject_split_is_not():
    from dms.data.datasets import random_split

    df = pd.DataFrame({"path": [f"{i}.png" for i in range(600)], "subject": [f"s{i % 10}" for i in range(600)], "source": "x", "split": "", "eye": 1, "yawn": -1, "emotion": -1, "cabin": -1})
    with pytest.raises(AssertionError):
        assert_no_leakage(random_split(df, seed=0))
    assert_no_leakage(subject_split(df, seed=0))


def test_evaluate_alerts_scoring():
    from dms.tools.evaluate_alerts import alert_times, score
    from dms.tools.labels import Segment

    segs = [Segment("a.mov", 10, 12.5, "eyes_closed"), Segment("a.mov", 30, 35, "look_away"), Segment("a.mov", 50, 50.5, "eyes_closed")]
    log = [{"t": 11.0, "alerts": ["microsleep"]}, {"t": 14.0, "alerts": ["drowsy"]}, {"t": 33.0, "alerts": ["eyes_off_road"]},
           {"t": 70.0, "alerts": ["eyes_off_road"]}, {"t": 80.0, "alerts": [], "yawns": 1}]
    r = score(segs, alert_times(log), tolerance=3.0)
    assert r["microsleep"]["events"] == 1 and r["microsleep"]["detected"] == 1  # 0.5 s closure is not required
    assert r["microsleep"]["delays"] == [1.0]
    assert r["eyes_off_road"]["detected"] == 1 and r["eyes_off_road"]["false_alarms"] == 1  # the one at t=70
    assert r["drowsy"]["false_alarms"] == 0  # follow-on alert after a microsleep is fine
    assert r["yawn"]["false_alarms"] == 1  # yawn counter rose with no labelled yawn


def test_same_file_name_in_train_and_test_is_not_one_subject(tmp_path):
    import cv2
    from dms.data.manifests import emotion_folders

    img = np.zeros((8, 8, 3), np.uint8)
    for split in ("train", "test"):
        for emo in ("angry", "happy", "sad"):
            d = tmp_path / split / emo
            d.mkdir(parents=True)
            for k in range(3):
                cv2.imwrite(str(d / f"im{k}.png"), img)  # identical names across splits, like FER2013
    df = emotion_folders(tmp_path, "fer")
    assert df.subject.nunique() == len(df) == 18
    assert_no_leakage(subject_split(df, seed=0))


# ---------------------------------------------------------------- audio / mishap / eye fusion
def test_audio_group_scorer():
    from dms.perception.audio import GroupScorer

    labels = ["Speech", "Snoring", "Smash, crash", "Skidding", "Ringtone"]
    sc = GroupScorer(labels, decay=0.5)
    g = sc.update(np.array([0.9, 0.0, 0.8, 0.1, 0.0], np.float32))
    assert g["impact"] == pytest.approx(0.8) and g["snoring"] == 0.0 and g["siren"] == 0.0  # missing classes -> 0
    g = sc.update(np.zeros(5, np.float32))
    assert g["impact"] == pytest.approx(0.4)  # decays instead of vanishing


def test_jolt_detector_fires_on_whole_frame_change_only():
    import cv2
    from dms.fusion.mishap import JoltDetector

    base = np.zeros((240, 320, 3), np.uint8)
    base[:] = (60, 70, 80)
    cv2.rectangle(base, (0, 170), (320, 240), (30, 30, 35), -1)  # dashboard
    cv2.rectangle(base, (250, 20), (310, 90), (200, 200, 200), -1)  # window
    jd, fired = JoltDetector(), []
    for i in range(60):
        f = base.copy()
        cv2.circle(f, (150 + int(5 * np.sin(i / 3)), 100), 50, (150, 160, 200), -1)  # head moving normally
        if i == 45:
            f = np.roll(np.roll(f, 70, axis=1), 40, axis=0)  # whole image shifts: camera shaken
        fired.append(jd.update(i / 15, f))
    assert fired[45] and sum(fired) <= 2  # the jolt frame (and possibly the frame after it)

    # not jolts: lights change (tunnel), the face disappears, a hand covers the lens
    jd = JoltDetector()
    face = base.copy()
    cv2.circle(face, (150, 100), 50, (150, 160, 200), -1)
    seq = [face] * 20 + [(face * 0.4).astype(np.uint8)] * 5 + [base] * 5 + [np.full_like(base, 20)] * 5
    assert not any(jd.update(i / 15, f) for i, f in enumerate(seq))


def _mishap_run(responses, evidence_t=1.0, until=40.0, ack_at=None):
    from dms.fusion.mishap import MishapDetector

    md, events, t = MishapDetector(), [], 0.0
    while t < until:
        jolt = abs(t - evidence_t) < 1e-6
        face, closed = responses(t)
        md.update(t, jolt=jolt, audio_groups=None, face=face, eyes_closed=closed)
        if ack_at is not None and abs(t - ack_at) < 1e-6:
            md.acknowledge(t)
        events += [k for _, k, _ in md.pop_events()]
        t = round(t + 0.1, 1)
    return events


def test_mishap_driver_ok_after_impact():
    assert _mishap_run(lambda t: (True, False)) == ["impact"]


def test_mishap_unresponsive_leads_to_sos_unless_acknowledged():
    gone = lambda t: (t < 1.0, False)  # face disappears right after the jolt
    assert _mishap_run(gone) == ["possible_accident", "sos"]
    assert _mishap_run(gone, ack_at=10.0) == ["possible_accident", "driver_ok"]


def test_mishap_needs_evidence():
    from dms.fusion.mishap import MishapDetector

    md = MishapDetector()
    for i in range(100):  # driver vanishes but nothing suggested an impact -> no accident alarm
        md.update(i / 10, jolt=False, audio_groups={"impact": 0.1}, face=False, eyes_closed=False)
    assert md.pop_events() == []


def test_audio_alerts_and_mishap_alerts():
    from dms.fusion.alerts import AlertManager
    from dms.fusion.risk import RiskOutput

    am, risk = AlertManager(), RiskOutput(0, "OK", 0, 0, 0)
    kw = dict(risk=risk, microsleep=False, closed_duration=0, long_distraction=False, vats=False, off_road_duration=0,
              yawns_5min=0, face=True, alertness=None, emotion_probs_affect=0.0, behaviours={})
    kinds = [a.kind for i in range(30) for a in am.update(i * 0.1, **kw, audio_groups={"snoring": 0.9, "siren": 0.8})]
    assert "snoring" in kinds and "siren" in kinds
    a = am.update(5.0, **kw, mishap_events=[(5.0, "possible_accident", "camera jolt")])
    assert a[0].kind == "possible_accident" and a[0].severity == "critical"


def test_eye_cnn_fusion():
    tr = EyeClosureTracker()
    tr.update(0.0, 0.30, PROF, p_closed=0.9)  # geometry says open, confident CNN says closed
    assert tr.last_closed
    tr.update(0.1, 0.05, PROF, p_closed=0.1)  # geometry says closed, confident CNN says open (e.g. looking down)
    assert not tr.last_closed
    tr.update(0.2, 0.05, PROF, p_closed=0.5)  # CNN unsure -> geometry decides
    assert tr.last_closed


def test_risk_uses_audio():
    rf = RiskFusion()
    out = rf.update(0.0, RiskInputs(audio_groups={"snoring": 0.95}))
    assert out.drowsiness > 0.5 and any("snoring" in r for r in out.reasons)


# ---------------------------------------------------------------- offline store / security
def test_event_store_offline_then_sync(tmp_path):
    import http.server
    import threading

    from dms.storage import EventStore, Syncer

    store = EventStore(tmp_path / "ev.db")
    sid = store.new_session()
    store.add_event(sid, "microsleep", "critical", t_session=9.0, detail="eyes closed 1.0s")
    store.save_trip(sid, {"safety_score": 80})
    assert store.counts()["events_unsynced"] == 1 and store.counts()["trips_unsynced"] == 1

    received = []

    class H(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            assert self.headers["Authorization"] == "Bearer secret"
            self.send_response(200)
            self.end_headers()

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    port = srv.server_address[1]
    sync = Syncer(store, f"http://127.0.0.1:{port}/upload", "secret")
    srv.server_close()  # "offline": nothing listening yet
    with pytest.raises(Exception):
        sync.sync_once()
    assert store.counts()["events_unsynced"] == 1  # data kept while offline

    srv = http.server.HTTPServer(("127.0.0.1", port), H)  # back online
    threading.Thread(target=srv.handle_request, daemon=True).start()
    assert sync.sync_once() == 2
    srv.server_close()
    assert received[0]["events"][0]["kind"] == "microsleep" and received[0]["trips"][0]["summary"]["safety_score"] == 80
    assert store.counts()["events_unsynced"] == 0
    with pytest.raises(ValueError):
        Syncer(store, "http://example.com/upload")  # plain http to a remote host is refused


def test_security_rules(tmp_path):
    from dms import security

    assert security.stream_url_allowed("rtsp://192.168.1.254/live")[0]
    assert security.stream_url_allowed("http://10.0.0.5:8080/video")[0]
    assert security.stream_url_allowed("0")[0]
    assert not security.stream_url_allowed("/etc/passwd")[0]
    assert not security.stream_url_allowed("file:///Users/me/private.mov")[0]
    assert not security.stream_url_allowed("ftp://x/y")[0]
    f = tmp_path / "clip.mp4"
    f.write_bytes(b"x")
    assert security.stream_url_allowed(str(f), allow_files=True)[0]
    # cross-site WebSocket hijacking and DNS rebinding
    assert security.origin_allowed("http://localhost:8000", "localhost:8000")
    assert not security.origin_allowed("https://evil.example", "localhost:8000")
    assert security.host_allowed("localhost:8000", "127.0.0.1")
    assert not security.host_allowed("evil.example:8000", "127.0.0.1")
    assert security.host_allowed("192.168.1.5:8000", "0.0.0.0")
    assert security.token_ok(None, None) and security.token_ok("abc", "abc") and not security.token_ok("abc", "abd")
    # model integrity
    m = tmp_path / "model.onnx"
    m.write_bytes(b"weights")
    security.write_sums([m], tmp_path / "SHA256SUMS")
    assert security.verify_weights(tmp_path / "SHA256SUMS") == []
    m.write_bytes(b"tampered")
    assert security.verify_weights(tmp_path / "SHA256SUMS")
    assert security.verify_weights() == []  # the shipped models match their checksums


def test_web_server_security_end_to_end():
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from starlette.websockets import WebSocketDisconnect

    from dms.web import server

    server.CFG.update(token=None, bind_host="127.0.0.1", allow_files=False, origins=set(), audio=False, mishap=False, models_info={})
    c = TestClient(server.app, base_url="http://localhost:8000")
    r = c.get("/")
    assert r.status_code == 200 and "Content-Security-Policy" in r.headers and r.headers["X-Frame-Options"] == "DENY"
    assert c.get("/", headers={"host": "evil.example:8000"}).status_code == 421  # DNS rebinding
    with pytest.raises(WebSocketDisconnect):  # another website trying to use the camera socket
        with c.websocket_connect("ws://localhost:8000/ws", headers={"origin": "https://evil.example"}) as ws:
            ws.receive_text()
    with c.websocket_connect("ws://localhost:8000/ws", headers={"origin": "http://localhost:8000"}) as ws:
        assert json.loads(ws.receive_text())["type"] == "ready"
        ws.send_text(json.dumps({"type": "stream_start", "url": "/etc/passwd"}))
        msg = json.loads(ws.receive_text())
        assert msg["type"] == "error" and "allowed" in msg["message"]

    server.CFG.update(token="s3cret", bind_host="0.0.0.0")
    c = TestClient(server.app, base_url="http://192.168.1.5:8000")
    assert c.get("/").status_code == 401
    assert c.get("/api/info").status_code == 401
    r = c.get("/?token=s3cret")
    assert r.status_code == 200 and "httponly" in r.headers["set-cookie"].lower()
    assert c.get("/api/info").status_code == 200  # cookie now authorises the browser
    server.CFG.update(token=None, bind_host="127.0.0.1")


def test_temperature_scaling_fixes_overconfidence():
    from dms.training.metrics import expected_calibration_error
    from dms.training.train_image import fit_temperature

    g = torch.Generator().manual_seed(0)
    y = torch.randint(0, 3, (3000,), generator=g)
    # an overconfident model: right 70% of the time but ~100% sure
    pred = torch.where(torch.rand(3000, generator=g) < 0.7, y, (y + 1) % 3)
    logits = torch.nn.functional.one_hot(pred, 3).float() * 12
    T = fit_temperature(logits, y)
    before = expected_calibration_error(torch.softmax(logits, -1).numpy(), y.numpy())
    after = expected_calibration_error(torch.softmax(logits / T, -1).numpy(), y.numpy())
    assert T > 2 and after < before / 3
    assert (torch.softmax(logits / T, -1).argmax(1) == pred).all()  # predictions unchanged


def test_sensitivity_presets_change_alert_timing():
    from dms.pipeline import SENSITIVITY

    def microsleeps(level, closed_s):
        tr = EyeClosureTracker(microsleep_s=SENSITIVITY[level]["microsleep_s"])
        for i in range(90):  # 3 s at 30 FPS, eyes closed from 1.0 s for `closed_s`
            t = i / 30
            tr.update(t, 0.05 if 1.0 <= t < 1.0 + closed_s else 0.30, PROF)
        return tr.microsleep_events

    assert microsleeps("normal", 1.2) == 1 and microsleeps("low", 1.2) == 0  # low tolerates a 1.2 s closure
    assert microsleeps("high", 0.9) == 1 and microsleeps("normal", 0.9) == 0
    assert SENSITIVITY["low"]["long_s"] > SENSITIVITY["normal"]["long_s"] > SENSITIVITY["high"]["long_s"]


def test_dataset_discovery_by_content(tmp_path):
    import cv2
    from dms.data.discover import discover

    img = np.zeros((16, 16, 3), np.uint8)

    def put(*parts, n=3):
        d = tmp_path.joinpath(*parts)
        d.mkdir(parents=True, exist_ok=True)
        for k in range(n):
            cv2.imwrite(str(d / f"{parts[-1].replace(' ', '_')}_{k}.png"), img)

    for sp in ("train", "test"):  # FER2013 (Kaggle new layout: datasets/<owner>/<slug>)
        for e in ("angry", "happy", "sad", "neutral", "surprise"):
            put("datasets", "msambare", "fer2013", sp, e)
    for e in ("anger", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"):  # AffectNet copy
        put("datasets", "someone", "affectnet-training-data", e)
    for sp in ("train", "test"):  # RAF-DB copy with numeric folders
        for i in range(1, 8):
            put("datasets", "x", "raf-db-dataset", "DATASET", sp, str(i))
    for f in ("yawn", "no_yawn", "Open", "Closed"):  # Yawn-Eye
        put("datasets", "y", "yawn-eye-dataset-new", "dataset_new", "train", f)
    for f in ("Drowsy", "Non Drowsy"):  # leaky DDD -> must be skipped
        put("datasets", "z", "driver-drowsiness-dataset-ddd", f)
    r = discover(tmp_path, tmp_path / "manifests")
    kinds = sorted(f.kind for f in r["found"])
    assert kinds == ["drowsyframes", "emotion", "emotion", "rafdb", "yawneye"]
    assert r["drowsy"] == []  # the DDD copy has no person IDs -> not used
    raf = pd.read_csv(next(p for p in r["face"] if "rafdb" in p))
    assert set(raf.emotion) == set(range(7)) and set(raf.split) == {"train", "test"}
    aff = pd.read_csv(next(p for p in r["face"] if "affectnet" in p))
    assert len(aff) == 7 * 3  # contempt is not one of our 7 classes
    ye = pd.read_csv(next(p for p in r["face"] if "yawn" in p))
    assert (ye.yawn >= 0).sum() == 6 and (ye.eye >= 0).sum() == 6


def test_discovery_of_official_affectnet_and_csv_layouts(tmp_path):
    import cv2
    from dms.data.discover import discover

    img = np.zeros((16, 16, 3), np.uint8)
    # official AffectNet layout: <split>/images/<id>.jpg + <split>/annotations/<id>_exp.npy
    for split, ids in (("train_set", range(0, 24)), ("val_set", range(100, 116))):
        (tmp_path / "datasets/mohammadmahoor/affectnetplus" / split / "images").mkdir(parents=True)
        (tmp_path / "datasets/mohammadmahoor/affectnetplus" / split / "annotations").mkdir(parents=True)
        for i in ids:
            base = tmp_path / "datasets/mohammadmahoor/affectnetplus" / split
            cv2.imwrite(str(base / "images" / f"{i}.jpg"), img)
            np.save(base / "annotations" / f"{i}_exp.npy", np.array(str(i % 9)))  # 7=contempt, 8=none -> dropped
    # CSV-labelled dataset
    d = tmp_path / "datasets/someone/emotion-table/imgs"
    d.mkdir(parents=True)
    names = ["happy", "sad", "angry", "neutral", "fear", "surprise"] * 3
    for k, n in enumerate(names):
        cv2.imwrite(str(d / f"img{k}.png"), img)
    pd.DataFrame({"image_path": [f"imgs/img{k}.png" for k in range(len(names))], "emotion": names}).to_csv(d.parent / "labels.csv", index=False)

    r = discover(tmp_path, tmp_path / "manifests")
    aff = pd.read_csv(next(p for p in r["face"] if "affectnet" in p))
    assert set(aff.emotion) == set(range(7)) and len(aff) == len([i for i in list(range(24)) + list(range(100, 116)) if i % 9 < 7])
    assert set(aff.split) == {"train", "test"}  # AffectNet's official validation set is reported as test (as in the literature)
    tab = pd.read_csv(next(p for p in r["face"] if "emo_emotion_table" in p))
    assert len(tab) == len(names)


def test_nthu_frames_need_person_ids(tmp_path):
    import cv2
    from dms.data.discover import discover

    img = np.zeros((16, 16, 3), np.uint8)
    with_ids = tmp_path / "with/datasets/x/nthu-ddd"
    for lab in ("drowsy", "notdrowsy"):
        (with_ids / lab).mkdir(parents=True)
        for s in range(1, 6):
            for k in range(3):
                cv2.imwrite(str(with_ids / lab / f"{s:03d}_glasses_sleepyCombination_{k}_{lab}.jpg"), img)
    r = discover(tmp_path / "with", tmp_path / "m1")
    df = pd.read_csv(r["drowsy"][0], dtype={"subject": str})
    assert df.subject.nunique() == 5 and set(df.drowsy) == {0, 1}

    no_ids = tmp_path / "without/datasets/y/driver-drowsiness-dataset-ddd"
    for lab in ("Drowsy", "Non Drowsy"):
        (no_ids / lab).mkdir(parents=True)
        for k in range(5):
            cv2.imwrite(str(no_ids / lab / f"frame{k}.png"), img)
    assert discover(tmp_path / "without", tmp_path / "m2")["drowsy"] == []  # leaky -> skipped


def test_rldd_video_layouts(tmp_path):
    from dms.data.sequences import find_videos

    for rel in ("Fold1_part1/01/0.mov", "Fold1_part1/01/5.MOV", "Fold1_part1/01/10.mp4",
                "byclass/alert/07.mp4", "byclass/drowsy/07.mp4", "flat/12_low_vigilant.avi", "flat/README.txt"):
        f = tmp_path / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"")
    got = sorted((s, l) for s, l, _ in find_videos(tmp_path))
    assert got == [("01", 0), ("01", 1), ("01", 2), ("07", 0), ("07", 2), ("12", 1)]


def test_affectnet_plus_json_validation_only_is_heldout(tmp_path):
    import cv2
    from dms.data.discover import discover, json_label

    assert json_label({"human-label": 1, "soft-label": [0.1, 0.7, 0, 0, 0, 0, 0.2, 0]}) == 1  # human label wins
    assert json_label({"soft-label": [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.1]}) == 6  # argmax -> anger
    assert json_label({"labels": {"expression": "happy"}}) == 1  # nested, by name
    assert json_label({"human-label": 7}) is None  # contempt is not one of our classes

    base = tmp_path / "datasets/mohammadmahoor/affectnetplus/AffeceNet+"
    (base / "train_set").mkdir(parents=True)
    (base / "train_set/readme.txt").write_text("training set available on request")
    for d in ("images", "annotations"):
        (base / "validation_set" / d).mkdir(parents=True)
    for i in range(21):
        cv2.imwrite(str(base / "validation_set/images" / f"{i}.jpg"), np.zeros((16, 16, 3), np.uint8))
        (base / "validation_set/annotations" / f"{i}.json").write_text(json.dumps({"human-label": i % 8, "soft-label": [0.125] * 8}))
    r = discover(tmp_path, tmp_path / "m")
    assert r["face"] == [] and len(r["face_test"]) == 1  # validation only -> held-out test, never trained on
    df = pd.read_csv(r["face_test"][0])
    assert set(df.split) == {"test"} and len(df) == 21 - 2  # labels 7 (contempt) dropped


def test_hybrid_temporal_model_and_binary(tmp_path):
    from dataclasses import asdict

    from dms.models.networks import DrowsinessHybrid, sequence_stats
    from dms.training.train_temporal import load_dir

    x = torch.randn(4, 300, 9)
    assert sequence_stats(x).shape == (4, 5 + 2 * 8)
    for nc in (3, 2):
        m = DrowsinessHybrid(in_dim=9, num_classes=nc).eval()
        assert m(x).shape == (4, nc)
        checkpoint.save(tmp_path / f"h{nc}.pt", m, "temporal", {"kind": "hybrid", "in_dim": 9, "num_classes": nc})
        m2, _ = checkpoint.load(tmp_path / f"h{nc}.pt")
        assert torch.allclose(m(x), m2(x), atol=1e-5)
    old = DrowsinessTransformer(in_dim=9).eval()  # checkpoints saved before the embed() refactor still load
    checkpoint.save(tmp_path / "old.pt", old, "temporal", {"kind": "transformer", "in_dim": 9})
    assert torch.allclose(old(x), checkpoint.load(tmp_path / "old.pt")[0](x), atol=1e-6)

    d = tmp_path / "seq"
    d.mkdir()
    for s in range(3):
        for lab in (0, 1, 2):
            np.savez(d / f"x_{s}_{lab}.npz", raw=np.random.rand(50, 9).astype(np.float32), label=lab, subject=f"s{s}",
                     profile=np.array(list(asdict(PROF).values()), dtype=np.float64))
    v = load_dir(d, binary=True)
    assert len(v) == 6 and sorted({x["y"] for x in v}) == [0, 1]  # low-vigilant dropped, drowsy -> 1


def test_risk_accepts_binary_temporal_model():
    out = RiskFusion().update(0.0, RiskInputs(alertness_probs=np.array([0.1, 0.9])))
    assert out.drowsiness > 0.4 and any("drowsy" in r for r in out.reasons)
