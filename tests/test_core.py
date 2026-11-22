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
