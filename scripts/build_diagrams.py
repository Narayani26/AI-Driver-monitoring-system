"""Engineering diagrams for the report (HTML boxes rendered to PNG with headless Chromium)."""
import subprocess
import tempfile
from pathlib import Path

OUT = Path("docs/report_assets").resolve()
CSS = """
body{margin:0;font:15px 'Liberation Sans',Arial,sans-serif;color:#111;background:#fff}
.d{display:inline-block;padding:22px 26px;background:#fff}
.t{font-weight:bold;font-size:17px;margin:0 0 14px}
.row{display:flex;gap:16px;align-items:center;justify-content:center}
.col{display:flex;flex-direction:column;gap:12px;align-items:center}
.b{border:2px solid #334155;border-radius:10px;padding:9px 13px;text-align:center;background:#f8fafc;min-width:120px}
.p{border-radius:50%;padding:20px 18px;border:2px solid #1d4ed8;background:#eff6ff;text-align:center;min-width:140px}
.s{border-top:2px solid #b45309;border-bottom:2px solid #b45309;padding:7px 14px;background:#fffbeb;text-align:center}
.e{border:2px solid #0f172a;padding:9px 14px;background:#e2e8f0;text-align:center;font-weight:bold}
.a{font-size:24px;color:#475569}
.small{font-size:12.5px;color:#334155}
.uc{border:2px solid #7c3aed;border-radius:999px;padding:8px 16px;background:#f5f3ff;text-align:center;font-size:14px}
.actor{text-align:center;font-weight:bold}
.sys{border:2px dashed #64748b;border-radius:14px;padding:14px;display:flex;flex-direction:column;gap:9px}
.tbl{border:2px solid #334155;border-radius:6px;min-width:210px;background:#fff}
.tbl .h{background:#1e3a8a;color:#fff;font-weight:bold;padding:6px 10px}
.tbl div.f{padding:3px 10px;font-size:13px;border-top:1px solid #e2e8f0;font-family:'Liberation Mono',monospace}
.st{border:2px solid #334155;border-radius:14px;padding:10px 16px;text-align:center;min-width:110px;font-weight:bold}
.ok{background:#dcfce7}.ca{background:#fef9c3}.wa{background:#ffedd5}.cr{background:#fee2e2}
.lane{display:grid;grid-template-columns:150px repeat(4,1fr);gap:0;font-size:13.5px}
.lane .hd{font-weight:bold;text-align:center;border-bottom:2px solid #334155;padding:6px}
.lane .c{padding:5px 8px;border-left:1px dashed #94a3b8;min-height:24px}
.lane .n{font-weight:bold;color:#1e3a8a;padding:5px 8px}
"""

D = {}
D["dfd0"] = """<div class='t'>Context diagram (DFD level 0)</div>
<div class='row'><div class='col'><div class='e'>Camera</div><div class='e'>Microphone</div></div>
<div class='col'><div class='a'>&rarr; frames<br>&rarr; audio</div></div>
<div class='p'><b>0</b><br>AI Driver<br>Monitoring System</div>
<div class='col'><div class='a'>&rarr; alerts, voice<br>&larr; feedback, KSS</div></div>
<div class='col'><div class='e'>Driver</div></div></div>
<div class='row' style='margin-top:16px'><div class='a'>&darr; trips, alerts, samples &nbsp;&nbsp; &uarr; approved model</div></div>
<div class='row' style='margin-top:10px'><div class='e'>Cloud back end (Supabase PostgreSQL)</div><div class='a'>&harr;</div><div class='e'>Administrator</div></div>"""
D["dfd1"] = """<div class='t'>Data-flow diagram (level 1)</div>
<div class='row'><div class='e'>Camera / Mic</div><div class='a'>&rarr;</div>
<div class='p'><b>1.0</b><br>Capture &amp;<br>decode</div><div class='a'>&rarr;</div>
<div class='p'><b>2.0</b><br>Perception<br><span class='small'>landmarks, CNNs, YOLO</span></div><div class='a'>&rarr;</div>
<div class='p'><b>3.0</b><br>Driver-state<br>estimation</div><div class='a'>&rarr;</div>
<div class='p'><b>4.0</b><br>Risk fusion<br>&amp; alerts</div><div class='a'>&rarr;</div><div class='e'>Driver</div></div>
<div class='row' style='margin-top:18px;gap:60px'>
<div class='col'><div class='a'>&uarr;&darr;</div><div class='s'>D1 Driver profile (calibration)</div></div>
<div class='col'><div class='a'>&darr;</div><div class='p'><b>5.0</b><br>Store &amp; sync</div><div class='a'>&darr;</div><div class='s'>D2 Local SQLite</div></div>
<div class='col'><div class='a'>&rarr;</div><div class='s'>D3 Cloud PostgreSQL</div><div class='a'>&darr;</div><div class='p'><b>6.0</b><br>Retrain &amp;<br>approve</div></div></div>"""
D["usecase"] = """<div class='t'>Use-case diagram</div>
<div class='row' style='align-items:stretch'>
<div class='col' style='justify-content:center'><div class='actor'>&#128100;<br>Driver</div><div style='height:30px'></div><div class='actor'>&#128187;<br>Device<br>(dash cam / phone)</div></div>
<div class='sys'><div style='text-align:center;font-weight:bold'>In-car application</div>
<div class='uc'>Start / stop monitoring</div><div class='uc'>Calibrate automatically</div><div class='uc'>Receive voice &amp; buzzer alerts</div>
<div class='uc'>Tap "Not true" / rate KSS</div><div class='uc'>View &amp; print trip report</div><div class='uc'>Confirm "I'm OK" after a possible crash</div></div>
<div class='sys'><div style='text-align:center;font-weight:bold'>Cloud back end</div>
<div class='uc'>Register / log in</div><div class='uc'>Add &amp; revoke devices</div><div class='uc'>View trips and alerts</div>
<div class='uc'>Give / withdraw data consent</div><div class='uc'>Upload events &amp; samples</div><div class='uc'>Download approved model</div>
<div class='uc'>Review data quality &amp; models</div></div>
<div class='col' style='justify-content:center'><div class='actor'>&#128100;<br>Administrator</div><div style='height:30px'></div><div class='actor'>&#9200;<br>Scheduler<br>(weekly retrain)</div></div></div>"""
D["sequence"] = """<div class='t'>Sequence of one video frame</div>
<div class='lane'><div class='hd'>Step</div><div class='hd'>Browser</div><div class='hd'>FastAPI session</div><div class='hd'>DriverMonitor</div><div class='hd'>Model thread</div>
<div class='n'>1</div><div class='c'>grab frame, JPEG</div><div class='c'></div><div class='c'></div><div class='c'></div>
<div class='n'>2</div><div class='c'>WebSocket send &rarr;</div><div class='c'>decode, run in thread pool</div><div class='c'></div><div class='c'></div>
<div class='n'>3</div><div class='c'></div><div class='c'>process(frame, t) &rarr;</div><div class='c'>face landmarks (zoom if needed)</div><div class='c'></div>
<div class='n'>4</div><div class='c'></div><div class='c'></div><div class='c'>schedule due models &rarr;</div><div class='c'>eye CNN, emotion, YOLO, pose, temporal</div>
<div class='n'>5</div><div class='c'></div><div class='c'></div><div class='c'>camera-move check, calibration</div><div class='c'>&larr; latest outputs</div>
<div class='n'>6</div><div class='c'></div><div class='c'></div><div class='c'>trackers &rarr; risk fusion &rarr; alerts</div><div class='c'></div>
<div class='n'>7</div><div class='c'></div><div class='c'>&larr; FrameResult; log alerts to SQLite</div><div class='c'></div><div class='c'></div>
<div class='n'>8</div><div class='c'>&larr; JSON result: overlay, gauge, voice</div><div class='c'></div><div class='c'></div><div class='c'></div>
<div class='n'>9</div><div class='c'>send next frame only now (no backlog)</div><div class='c'></div><div class='c'></div><div class='c'></div></div>"""
D["er"] = """<div class='t'>Entity-relationship diagram (cloud database)</div>
<div class='row' style='align-items:flex-start'>
<div class='tbl'><div class='h'>users</div><div class='f'>id PK</div><div class='f'>email UNIQUE</div><div class='f'>name</div><div class='f'>password_hash</div><div class='f'>role</div><div class='f'>consent_training</div><div class='f'>consent_at</div></div>
<div class='col'><div class='a'>1 &rarr; N</div></div>
<div class='tbl'><div class='h'>devices</div><div class='f'>id PK</div><div class='f'>user_id FK</div><div class='f'>name, kind</div><div class='f'>key_hash UNIQUE</div><div class='f'>key_prefix</div><div class='f'>last_seen, revoked</div></div>
<div class='col'><div class='a'>1 &rarr; N</div></div>
<div class='col'>
<div class='tbl'><div class='h'>trips</div><div class='f'>id PK (session)</div><div class='f'>user_id FK, device_id FK</div><div class='f'>t_start, t_end</div><div class='f'>summary JSON, kss</div></div>
<div class='tbl'><div class='h'>events</div><div class='f'>id PK</div><div class='f'>user_id FK, device_id FK</div><div class='f'>trip_id, t_wall, kind</div><div class='f'>severity, payload, feedback</div></div>
<div class='tbl'><div class='h'>samples</div><div class='f'>id PK</div><div class='f'>user_id FK, device_id FK</div><div class='f'>label, label_source</div><div class='f'>data JSON, quality, accepted</div></div></div>
<div class='col'><div class='tbl'><div class='h'>model_versions</div><div class='f'>id PK</div><div class='f'>kind, version</div><div class='f'>status</div><div class='f'>sha256, path</div><div class='f'>metrics, data</div></div></div></div>"""
D["levels"] = """<div class='t'>Risk-level state machine with hysteresis</div>
<div class='row'><div class='st ok'>OK</div><div class='a'>&rarr;<br><span class='small'>R &ge; 30</span></div>
<div class='st ca'>CAUTION</div><div class='a'>&rarr;<br><span class='small'>R &ge; 55</span></div>
<div class='st wa'>WARNING</div><div class='a'>&rarr;<br><span class='small'>R &ge; 80</span></div><div class='st cr'>CRITICAL</div></div>
<div class='row' style='margin-top:12px'><div class='small'>&larr; back down only when R falls 10 points below the entry threshold (CRITICAL&rarr;WARNING at R &lt; 70, WARNING&rarr;CAUTION at R &lt; 45, CAUTION&rarr;OK at R &lt; 20)</div></div>
<div class='row' style='margin-top:12px'><div class='b'>microsleep or head drop &rArr; CRITICAL immediately</div><div class='b'>learned models only &rArr; at most OK/green (R &le; 28)</div><div class='b'>"Not true" &rArr; evidence reset, back to OK</div></div>"""
D["mishap"] = """<div class='t'>Possible-accident (mishap) state machine</div>
<div class='row'><div class='st ok'>idle</div><div class='a'>&rarr;<br><span class='small'>jolt or crash/skid<br>sound &ge; 0.3</span></div>
<div class='st ca'>checking<br><span class='small'>(silent, 10 s)</span></div><div class='a'>&rarr;<br><span class='small'>driver responsive<br>+ strong evidence</span></div>
<div class='st wa'>impact / crash-sound<br>notice</div></div>
<div class='row' style='margin-top:14px'><div class='a'>&darr; <span class='small'>no face or eyes closed for 3 s</span></div></div>
<div class='row'><div class='st cr'>possible accident<br><span class='small'>15 s countdown</span></div><div class='a'>&rarr;<br><span class='small'>no response</span></div><div class='st cr'>SOS raised</div>
<div class='a'>&nbsp;&nbsp;|&nbsp;&nbsp;</div><div class='st ok'>cleared<br><span class='small'>"I'm OK" pressed</span></div></div>"""
D["deploy"] = """<div class='t'>Deployment diagram</div>
<div class='row' style='align-items:stretch'>
<div class='sys'><b style='text-align:center'>In the vehicle</b><div class='b'>Laptop / mini PC / Raspberry Pi<br><span class='small'>Python app or Docker image, systemd service</span></div>
<div class='b'>Camera: webcam, USB IR camera,<br>Wi-Fi dash cam (RTSP), phone (IP camera)</div><div class='b'>Speaker + browser UI (optional)</div><div class='s'>SQLite event store (offline)</div></div>
<div class='col'><div class='a'>&rarr; HTTPS<br><span class='small'>device key</span><br>&larr; model</div></div>
<div class='sys'><b style='text-align:center'>Cloud</b><div class='b'>FastAPI back end (Docker / VM)<br><span class='small'>auth, ingest, dashboard, registry</span></div>
<div class='b'>Weekly retraining job<br><span class='small'>systemd timer</span></div><div class='s'>Supabase PostgreSQL 16<br><span class='small'>Row Level Security, TLS</span></div></div>
<div class='col'><div class='a'>&larr; browser</div></div>
<div class='sys'><b style='text-align:center'>Users</b><div class='b'>Drivers: trips, ratings, consent</div><div class='b'>Administrator: data quality, models</div></div></div>"""
D["retrain"] = """<div class='t'>Retraining and safe-deployment flow</div>
<div class='col'><div class='b'>Driver rates an alert (&#10003; / &#10007;) or gives KSS 1&ndash;9</div><div class='a'>&darr;</div>
<div class='b'>Consent? &nbsp;No &rArr; nothing is kept &nbsp;|&nbsp; Yes &rArr; 30 s of numeric features + label</div><div class='a'>&darr;</div>
<div class='b'>Quality gate: face &ge; 90%, frontal &ge; 70%, same driver, calibrated, moving &nbsp;&rArr; accepted / rejected (reason kept)</div><div class='a'>&darr;</div>
<div class='b'>Weekly: export split by driver &rarr; enough drivers? (&ge; 5 train, &ge; 2 test) &nbsp;No &rArr; skip</div><div class='a'>&darr;</div>
<div class='b'>Train candidate on UTA-RLDD + real windows</div><div class='a'>&darr;</div>
<div class='b'>Test candidate AND current model on held-out real drivers</div><div class='a'>&darr;</div>
<div class='row'><div class='st ok'>macro-F1 not lower AND false alarms not higher &rArr; APPROVED</div><div class='st cr'>otherwise &rArr; REJECTED (kept for audit)</div></div><div class='a'>&darr;</div>
<div class='b'>Devices download the approved model, verify SHA-256, keep the previous one for rollback</div></div>"""


def main():
    d = Path(tempfile.mkdtemp())
    for name, inner in D.items():
        (d / f"{name}.html").write_text(f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body><div class='d'>{inner}</div></body></html>")
    js = d / "r.js"
    js.write_text("const { chromium } = require('playwright');(async()=>{const b=await chromium.launch();const p=await b.newPage({viewport:{width:1500,height:900},deviceScaleFactor:2});"
                  + "".join(f"await p.goto('file://{d / n}.html');await (await p.$('.d')).screenshot({{path:'{OUT / ('diag_' + n + '.png')}'}});" for n in D)
                  + "await b.close();})();")
    subprocess.run(["node", str(js)], check=True)
    print("diagrams:", ", ".join(D))


if __name__ == "__main__":
    main()
