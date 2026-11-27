"use strict";
// ---------------------------------------------------------------- state
const $ = (id) => document.getElementById(id);
const S = {
  ws: null, ready: false, src: "webcam", running: false, inflight: false,
  t0: 0, lastT: -1, stream: null, fileURL: null, streamFrame: null,
  hist: [], alerts: [], audio: null, speakingSev: -1, lastLevel: "OK", report: null,
};
const SEV = { info: 0, caution: 1, warning: 2, critical: 3 };
const LVL_COLOR = { OK: "#3fb950", CAUTION: "#e3b341", WARNING: "#f0883e", CRITICAL: "#f85149" };
const EMO = ["neutral", "happy", "sad", "surprise", "fear", "disgust", "anger"];
const video = $("video"), ai = $("ai-canvas"), rawCanvas = $("raw-canvas");
const sendCanvas = document.createElement("canvas");

// ---------------------------------------------------------------- socket
function connect() {
  const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  ws.binaryType = "arraybuffer";
  S.ws = ws;
  setConn("loading models…", false);
  ws.onmessage = async (ev) => {
    if (ev.data instanceof ArrayBuffer) { // stream mode: raw frame precedes its result
      S.streamFrame = await createImageBitmap(new Blob([ev.data], { type: "image/jpeg" }));
      rawCanvas.width = S.streamFrame.width; rawCanvas.height = S.streamFrame.height;
      rawCanvas.getContext("2d").drawImage(S.streamFrame, 0, 0);
      return;
    }
    const m = JSON.parse(ev.data);
    if (m.type === "ready") {
      S.ready = true; setConn("connected", true); fillModels(m.models);
      S.features = m.features || {};
      if (S.features.sensitivity) $("sens-select").value = S.features.sensitivity;
      $("opt-audio-wrap").classList.toggle("hidden", !S.features.audio);
      $("audio-card").classList.toggle("hidden", !S.features.audio);
    }
    else if (m.type === "audio") onAudio(m);
    else if (m.type === "status") { $("src-label").textContent = m.message; }
    else if (m.type === "result") onResult(m);
    else if (m.type === "report") showReport(m);
    else if (m.type === "stream_ended") { if (S.running && S.src === "stream") stop(true); }
    else if (m.type === "error") { alert(m.message); if (S.src === "stream") stop(false); }
  };
  ws.onclose = () => { S.ready = false; S.inflight = false; setConn("disconnected – retrying", false); setTimeout(connect, 1500); };
}
function setConn(text, on) { const c = $("conn"); c.textContent = text; c.className = "pill " + (on ? "pill-on" : "pill-off"); }
function send(obj) { if (S.ws && S.ws.readyState === 1) S.ws.send(JSON.stringify(obj)); }

// ---------------------------------------------------------------- capture loop (webcam / file)
function startLoop() { if (!S.looping) { S.looping = true; captureLoop(); } }
function captureLoop() {
  if (!S.running || S.src === "stream") { S.looping = false; return; }
  requestAnimationFrame(captureLoop);
  if (S.inflight || !S.ready || video.readyState < 2 || video.videoWidth === 0) return;
  if (S.src === "file" && (video.paused || video.ended)) return;
  const t = S.src === "file" ? video.currentTime : (performance.now() - S.t0) / 1000;
  if (S.src === "file" && t < S.lastT - 0.5) { send({ type: "reset", keep_profile: true }); S.hist = []; S.lastT = -1; } // seeked back
  if (t <= S.lastT) return; // same video frame as last time
  S.lastT = t;
  const w = Math.min(640, video.videoWidth), h = Math.round(video.videoHeight * w / video.videoWidth);
  sendCanvas.width = w; sendCanvas.height = h;
  sendCanvas.getContext("2d").drawImage(video, 0, 0, w, h);
  S.inflight = true;
  sendCanvas.toBlob(async (blob) => {
    const jpg = new Uint8Array(await blob.arrayBuffer());
    const buf = new Uint8Array(8 + jpg.length);
    new DataView(buf.buffer).setFloat64(0, t, true);
    buf.set(jpg, 8);
    if (S.ws.readyState === 1) S.ws.send(buf); else S.inflight = false;
  }, "image/jpeg", 0.8);
}

// ---------------------------------------------------------------- results
function onResult(r) {
  S.inflight = false;
  const img = S.src === "stream" ? S.streamFrame : sendCanvas;
  if (img) drawOverlay(img, r);
  $("perf").textContent = `${r.proc_ms.toFixed(0)} ms/frame · ${r.fps ? r.fps.toFixed(0) : "–"} FPS`;

  setModeBadge(r);
  const cal = $("calib");
  if (r.calibrating != null) {
    cal.classList.remove("hidden"); $("calib-bar").style.width = (r.calibrating * 100).toFixed(0) + "%";
    $("calib-text").textContent = r.camera_moved || S.recentMove ? "Camera moved — re-learning the driver's position…"
      : r.mode === "none" ? "Looking for the driver… make sure the camera can see the driver's head"
      : r.mode === "side" ? "Side view: learning the driver's normal head position — just drive normally"
      : "Learning the driver's normal position — just drive normally";
  }
  else cal.classList.add("hidden");

  updateStats(r);
  if (r.risk) {
    S.hist.push({ t: r.t, risk: r.risk.index, d: r.risk.drowsiness * 100, x: r.risk.distraction * 100, a: r.risk.affect * 100 });
    while (S.hist.length && r.t - S.hist[0].t > 60) S.hist.shift();
    drawChart($("chart"), S.hist, 60);
  }
  for (const a of r.alerts || []) onAlert(a);
  if (r.mishap) onMishap(r.mishap);
}

// ---------------------------------------------------------------- possible accident
function onMishap(m) {
  const box = $("sos");
  if (m.phase === "possible_accident") {
    box.classList.remove("hidden");
    $("sos-title").textContent = "Possible accident!";
    $("sos-sub").textContent = `Are you OK? (${m.evidence})`;
    $("sos-count").textContent = Math.ceil(m.countdown);
    $("sos-note").textContent = "If nobody responds, your emergency contact will be alerted.";
    $("sos-ok").textContent = "I'm OK";
  } else if (m.phase === "sos") {
    box.classList.remove("hidden");
    $("sos-title").textContent = "Emergency SOS raised";
    $("sos-sub").textContent = m.evidence;
    $("sos-count").textContent = "SOS";
    $("sos-note").textContent = "Logged locally. In the vehicle app this calls/messages your emergency contact with your location.";
    $("sos-ok").textContent = "Dismiss";
  } else if (!box.classList.contains("hidden") && (m.phase === "idle" || m.phase === "cleared")) {
    box.classList.add("hidden");
  }
}
$("sos-ok").onclick = () => { send({ type: "ack", t: S.lastT }); $("sos").classList.add("hidden"); };

// ---------------------------------------------------------------- audio (opt-in; analysed in memory, never recorded)
const AUDIO_GROUPS = { snoring: "snoring", distress: "shouting", impact: "crash", skid: "skid", horn: "horn", siren: "siren", phone: "phone ring", baby: "baby crying" };
function onAudio(m) {
  const lvl = Math.max(0, Math.min(1, (m.rms_db + 60) / 60));
  $("b-audio").style.width = (lvl * 100).toFixed(0) + "%";
  $("v-audio").textContent = m.rms_db.toFixed(0) + " dB";
  $("audio-state").textContent = "listening";
  $("audio-groups").innerHTML = Object.entries(AUDIO_GROUPS).map(([k, label]) => {
    const v = (m.groups || {})[k] || 0;
    const cls = v >= 0.5 ? (["snoring", "impact", "skid"].includes(k) ? "chip on bad" : "chip on") : "chip";
    return `<span class="${cls}" title="${(v * 100).toFixed(0)}%">${label}</span>`;
  }).join("");
  $("audio-top").textContent = (m.top || []).map(([n, p]) => `${n} ${(p * 100).toFixed(0)}%`).join(" · ");
}
const A = { ctx: null, node: null, mic: null, mediaSrc: null, buf: [], n: 0 };
async function startAudio() {
  if (!S.features || !S.features.audio || !$("opt-audio").checked || A.node) return;
  try {
    if (!A.ctx) {
      A.ctx = new AudioContext({ sampleRate: 16000 });
      await A.ctx.audioWorklet.addModule("/static/audio-worklet.js");
    }
    await A.ctx.resume();
    let src;
    if (S.src === "file") {
      if (!A.mediaSrc) { A.mediaSrc = A.ctx.createMediaElementSource(video); A.mediaSrc.connect(A.ctx.destination); }
      src = A.mediaSrc;
    } else if (S.src === "webcam") {
      A.mic = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: false, autoGainControl: false } });
      src = A.ctx.createMediaStreamSource(A.mic);
    } else return; // network streams: audio not supported yet
    A.node = new AudioWorkletNode(A.ctx, "forwarder");
    A.node.port.onmessage = (e) => sendAudio(e.data);
    src.connect(A.node);
    A.src = src;
    $("audio-state").textContent = "listening";
  } catch (e) {
    $("audio-state").textContent = "unavailable: " + e.message;
  }
}
function stopAudio() {
  if (A.node) { try { A.src.disconnect(A.node); } catch (e) { /* already disconnected */ } A.node.port.onmessage = null; A.node = null; }
  if (A.mic) { A.mic.getTracks().forEach((t) => t.stop()); A.mic = null; }
  $("audio-state").textContent = "off";
}
function sendAudio(chunk) {
  if (!S.running || !S.ws || S.ws.readyState !== 1) return;
  const t = S.src === "file" ? video.currentTime : (performance.now() - S.t0) / 1000;
  const buf = new ArrayBuffer(12 + chunk.length * 4);
  new Uint8Array(buf, 0, 4).set([65, 85, 68, 48]); // "AUD0"
  new DataView(buf).setFloat64(4, t, true);
  new Float32Array(buf, 12).set(chunk);
  S.ws.send(buf);
}
$("opt-audio").onchange = (e) => { if (S.running) (e.target.checked ? startAudio() : stopAudio()); };

function drawOverlay(img, r) {
  ai.width = r.w; ai.height = r.h;
  const g = ai.getContext("2d");
  g.drawImage(img, 0, 0, r.w, r.h);
  const lvl = r.risk ? r.risk.level : "OK", col = LVL_COLOR[lvl];
  const lw = Math.max(2, r.w / 320);
  if (r.face_box) {
    const [x0, y0, x1, y1] = r.face_box;
    g.strokeStyle = col; g.lineWidth = lw; g.strokeRect(x0, y0, x1 - x0, y1 - y0);
  }
  if (r.eyes) {
    const closed = !!r.eye_closed;
    g.strokeStyle = closed ? "#f85149" : "#3fb950"; g.lineWidth = lw;
    for (const eye of r.eyes) poly(g, eye);
  }
  if (r.mouth) { g.strokeStyle = "rgba(88,166,255,.8)"; g.lineWidth = lw * 0.7; poly(g, r.mouth); }
  if (r.head_pts) { // side view: head keypoints (nose, eyes, ears, shoulders)
    g.fillStyle = r.nodding ? "#ff5a5a" : "#4ade80";
    for (const [x, y] of r.head_pts) { g.beginPath(); g.arc(x, y, lw * 2.2, 0, 2 * Math.PI); g.fill(); }
  }
  for (const d of r.detections || []) {
    const [x0, y0, x1, y1] = d.box;
    g.setLineDash(d.active ? [] : [6, 4]);
    g.strokeStyle = d.active ? "#f85149" : "#8b98a9"; g.lineWidth = lw;
    g.strokeRect(x0, y0, x1 - x0, y1 - y0);
    g.setLineDash([]);
    label(g, `${d.label} ${(d.conf * 100).toFixed(0)}%${d.active ? "" : " (ignored)"}`, x0, Math.max(y0 - 4, 14), d.active ? "#f85149" : "#8b98a9", r.w);
  }
  // banner
  const fs = Math.max(13, r.w / 34);
  g.fillStyle = "rgba(0,0,0,.55)"; g.fillRect(0, 0, r.w, fs * 1.7);
  g.fillStyle = col; g.fillRect(0, 0, 6, fs * 1.7);
  g.font = `700 ${fs}px system-ui, sans-serif`; g.fillStyle = col;
  const head = r.calibrating != null ? "CALIBRATING" : (r.risk ? `${lvl} · risk ${r.risk.index.toFixed(0)}` : "");
  g.fillText(head, 14, fs * 1.2);
  g.font = `${fs * 0.75}px system-ui, sans-serif`; g.fillStyle = "#e6edf3";
  const right = r.face ? (r.emotion ? `${r.emotion} ${(r.emotion_conf * 100).toFixed(0)}%` : "") : r.mode === "side" ? "side view" : "no driver";
  g.fillText(right, r.w - g.measureText(right).width - 10, fs * 1.15);
  // live timers
  const timers = [];
  if (r.eye_closed_s > 0.3) timers.push(`eyes closed ${r.eye_closed_s.toFixed(1)}s`);
  if (r.off_road_s > 0.5) timers.push(`looking away ${r.off_road_s.toFixed(1)}s`);
  timers.forEach((s, i) => label(g, s, 10, r.h - 12 - i * fs * 1.2, "#f0883e", r.w));
  $("ai-stage").className = "stage lvl-" + lvl;
}
function poly(g, pts) { g.beginPath(); pts.forEach(([x, y], i) => (i ? g.lineTo(x, y) : g.moveTo(x, y))); g.closePath(); g.stroke(); }
function label(g, text, x, y, color, W) {
  const fs = Math.max(12, W / 45); g.font = `600 ${fs}px system-ui, sans-serif`;
  const w = g.measureText(text).width;
  g.fillStyle = "rgba(0,0,0,.65)"; g.fillRect(x - 3, y - fs, w + 6, fs * 1.3);
  g.fillStyle = color; g.fillText(text, x, y);
}

function setText(id, text, cls) { const el = $(id); el.textContent = text; if (cls !== undefined) el.className = el.className.split(" ")[0] + " " + cls; }
function updateStats(r) {
  const rk = r.risk;
  if (rk) {
    const col = LVL_COLOR[rk.level];
    $("gauge").style.background = `conic-gradient(${col} ${rk.index * 3.6}deg, var(--panel2) 0deg)`;
    $("risk-num").textContent = rk.index.toFixed(0);
    $("risk-level").textContent = rk.level; $("risk-level").style.color = col;
    for (const [k, v] of [["drowsy", rk.drowsiness], ["distract", rk.distraction], ["affect", rk.affect]]) {
      $("b-" + k).style.width = (v * 100).toFixed(0) + "%"; $("v-" + k).textContent = (v * 100).toFixed(0);
    }
    $("reasons").innerHTML = rk.reasons.map((x) => `<li>${esc(x)}</li>`).join("");
  } else if (r.calibrating != null) {
    $("risk-num").textContent = "…"; $("risk-level").textContent = "CALIBRATING";
  }
  const f = r.features;
  if (r.mode === "side") {
    const c = r.eye_cnn_closed;
    setText("s-eyes", r.eyes_measurable ? (r.eye_closed ? "CLOSED" : "open") : "side view", r.eye_closed ? "v bad" : r.eyes_measurable ? "v good" : "v");
    $("s-eyes-sub").textContent = c != null ? `visible eye · CNN closed ${(c * 100).toFixed(0)}%` : "tracking the head from the side";
  } else if (!r.face) { setText("s-eyes", r.mode === "none" ? "no driver" : "no face", "v warn"); $("s-eyes-sub").textContent = ""; }
  else if (r.eyes_measurable === false) {
    setText("s-eyes", "not judged", "v");
    $("s-eyes-sub").textContent = "head tilted/turned: eye shape unreliable, attention tracking instead";
  } else if (r.eye_openness != null) {
    const closed = !!r.eye_closed;
    setText("s-eyes", closed ? "CLOSED" : "open", closed ? "v bad" : "v good");
    $("s-eyes-sub").textContent = `openness ${(r.eye_openness * 100).toFixed(0)}% · EAR ${f.ear?.toFixed(3)}` +
      (r.eye_cnn_closed != null ? ` · CNN closed ${(r.eye_cnn_closed * 100).toFixed(0)}%` : "");
  } else { setText("s-eyes", "…", "v"); $("s-eyes-sub").textContent = f.ear != null ? `EAR ${f.ear.toFixed(3)}` : ""; }
  const pc = (r.perclos || 0) * 100;
  setText("s-perclos", pc.toFixed(1) + "%", pc >= 15 ? "v bad" : pc >= 8 ? "v warn" : "v good");
  setText("s-blinks", r.blink_rate != null ? r.blink_rate.toFixed(0) : "–", "v");
  setText("s-yawns", String(r.yawns ?? 0), r.yawns >= 3 ? "v bad" : r.yawns >= 1 ? "v warn" : "v good");
  $("s-mouth").textContent = f.jaw_open_bs != null ? `jaw open ${(f.jaw_open_bs * 100).toFixed(0)}%` : (f.mar != null ? `MAR ${f.mar.toFixed(2)}` : "");
  if (r.calibrating != null) setText("s-attn", "…", "v");
  else if (r.mode === "side") setText("s-attn", r.nodding ? "HEAD DROPPING" : (r.head_drop ?? 0) > 0.18 ? "head lowering" : "head up",
    r.nodding ? "v bad" : (r.head_drop ?? 0) > 0.18 ? "v warn" : "v good");
  else setText("s-attn", r.off_road ? `away ${r.off_road_s.toFixed(1)}s` : "on road", r.off_road ? (r.off_road_s > 2 ? "v bad" : "v warn") : "v good");
  $("s-pose").textContent = r.mode === "side" ? (r.head_drop != null ? `head drop ${r.head_drop.toFixed(2)} · nods ${r.nods ?? 0}` : "side view")
    : f.yaw != null ? `yaw ${f.yaw.toFixed(0)}° · pitch ${f.pitch.toFixed(0)}°` : "";
  const beh = Object.entries(r.behaviours || {}).filter(([, v]) => v >= 0.5);
  if (r.mode === "none" || r.calibrating != null) setText("s-beh", beh.length ? beh.map(([k]) => k.replace("_", " ")).join(", ") : "–", beh.length ? "v bad" : "v");
  else setText("s-beh", beh.length ? beh.map(([k]) => k.replace("_", " ")).join(", ") : "safe", beh.length ? "v bad" : "v good");
  $("s-beh-sub").textContent = r.alertness ? `temporal model: ${r.alertness.replace("_", " ")}` : `${(r.detections || []).length} object(s) seen`;
  // emotion bars
  const p = r.emotion_probs;
  $("emo-top").textContent = p ? `${r.emotion} (${(r.emotion_conf * 100).toFixed(0)}%)` : (r.face ? "model not loaded" : r.mode === "side" ? "needs a frontal face" : "no face");
  $("emo-bars").innerHTML = EMO.map((e) => {
    const v = p ? p[e] : 0;
    return `<div class="row ${p && e === r.emotion ? "top" : ""}"><span>${e}</span><div class="hbar"><div style="width:${(v * 100).toFixed(0)}%"></div></div><b>${(v * 100).toFixed(0)}%</b></div>`;
  }).join("");
}
function setModeBadge(r) {
  const b = $("mode-badge");
  if (r.camera_moved) S.recentMove = true;
  if (r.calibrating == null) S.recentMove = false;
  let txt, cls;
  if (S.recentMove) [txt, cls] = ["camera moved · re-learning", "warn"];
  else if (r.calibrating != null) [txt, cls] = [r.mode === "none" ? "looking for driver" : `learning position · ${r.mode === "side" ? "side view" : "front view"}`, "info"];
  else if (r.primary === "side") [txt, cls] = [r.mode === "side" ? "side view · monitoring" : "side view · driver turned", r.mode === "side" ? "ok" : "warn"];
  else [txt, cls] = [r.face ? "front view · monitoring" : "front view · face not visible", r.face ? "ok" : "warn"];
  b.textContent = txt; b.className = "badge " + cls;
}
function esc(s) { return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }

// ---------------------------------------------------------------- alerts: feed, toast, sound, voice
function onAlert(a) {
  S.alerts.unshift(a);
  const ul = $("alerts");
  if (ul.querySelector(".empty")) ul.innerHTML = "";
  const li = document.createElement("li");
  li.className = a.severity;
  li.innerHTML = `<span class="t">${fmtTime(a.t)}</span><div class="m">${esc(a.message)}${a.detail ? ` <span class="muted">(${esc(a.detail)})</span>` : ""}</div><div class="g">💡 ${esc(a.suggestion)}</div>`;
  if (S.features && S.features.rating && SEV[a.severity] >= 1) {
    // driver feedback: counts false alarms, and (if sharing is on) labels training data
    const fb = document.createElement("div");
    fb.className = "fb";
    fb.innerHTML = '<span class="muted small">Was this right?</span> <button class="btn tiny" data-v="correct">✓ yes</button> <button class="btn tiny" data-v="false_alarm">✗ no</button>';
    fb.querySelectorAll("button").forEach((b) => (b.onclick = () => {
      send({ type: "feedback", t: a.t, kind: a.kind, verdict: b.dataset.v });
      fb.innerHTML = `<span class="muted small">${b.dataset.v === "correct" ? "Thanks: marked right" : "Thanks: marked as a false alarm"}</span>`;
    }));
    li.appendChild(fb);
  }
  ul.prepend(li);
  while (ul.children.length > 50) ul.lastChild.remove();
  $("alert-count").textContent = `${S.alerts.length} total`;
  if (SEV[a.severity] >= 1) toast(a);
  if ($("opt-sound").checked) playSound(a.sound);
  if ($("opt-voice").checked && SEV[a.severity] >= 2) speak(a); // cautions and notices: tone + message only, no voice
}
let toastTimer = null;
function toast(a) {
  const t = $("toast");
  t.className = "toast " + a.severity;
  t.querySelector(".toast-title").textContent = a.message;
  t.querySelector(".toast-sub").textContent = a.suggestion;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.add("hidden"), a.severity === "critical" ? 6000 : 4500);
}
function ensureAudio() {
  if (!S.audio) { try { S.audio = new (window.AudioContext || window.webkitAudioContext)(); } catch (e) { S.audio = null; } }
  if (S.audio && S.audio.state === "suspended") S.audio.resume();
}
function tone(freq, start, dur, type = "sine", vol = 0.25) {
  const ctx = S.audio; if (!ctx) return;
  const o = ctx.createOscillator(), g = ctx.createGain();
  o.type = type; o.frequency.value = freq;
  g.gain.setValueAtTime(0, ctx.currentTime + start);
  g.gain.linearRampToValueAtTime(vol, ctx.currentTime + start + 0.01);
  g.gain.linearRampToValueAtTime(0, ctx.currentTime + start + dur);
  o.connect(g).connect(ctx.destination); o.start(ctx.currentTime + start); o.stop(ctx.currentTime + start + dur + 0.02);
}
function playSound(kind) {
  ensureAudio();
  if (kind === "alarm") for (let i = 0; i < 6; i++) tone(i % 2 ? 660 : 990, i * 0.18, 0.16, "square", 0.22);
  else if (kind === "chime") { tone(660, 0, 0.18); tone(880, 0.2, 0.25); }
  else if (kind === "soft") tone(520, 0, 0.2, "sine", 0.15);
}
function speak(a) {
  if (!("speechSynthesis" in window)) return;
  const sev = SEV[a.severity];
  if (speechSynthesis.speaking && sev <= S.speakingSev) return; // don't interrupt something more important
  speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(sev >= 2 ? `${a.message} ${a.suggestion}` : a.message);
  applyVoice(u);
  u.onend = () => (S.speakingSev = -1);
  S.speakingSev = sev;
  speechSynthesis.speak(u);
}
// ---------------------------------------------------------------- alert voice (female by default, choosable)
// Best-sounding female voices first. Which exist depends on the OS/browser:
// Edge "Natural" voices, Chrome's Google voices, macOS (Samantha, Veena = Indian English, Karen, ...), Windows (Aria, Jenny, Zira).
const FEMALE_HINTS = ["Aria Online (Natural)", "Jenny Online (Natural)", "Neerja Online (Natural)", "Sonia Online (Natural)",
  "Samantha (Enhanced)", "Ava (Premium)", "Ava (Enhanced)", "Zoe (Premium)", "Zoe (Enhanced)", "Google UK English Female",
  "Veena", "Samantha", "Ava", "Karen", "Moira", "Tessa", "Fiona", "Victoria", "Serena", "Google US English", "Microsoft Zira", "Zira", "female"];
function femaleRank(v) {
  const i = FEMALE_HINTS.findIndex((h) => v.name.toLowerCase().includes(h.toLowerCase()));
  return i < 0 ? 999 : i;
}
let VOICES = [];
function loadVoices() {
  if (!("speechSynthesis" in window)) return;
  VOICES = speechSynthesis.getVoices().filter((v) => v.lang.toLowerCase().startsWith("en"))
    .sort((a, b) => femaleRank(a) - femaleRank(b) || a.name.localeCompare(b.name));
  const sel = $("voice-select");
  let saved = null;
  try { saved = localStorage.getItem("dms_voice"); } catch (e) { /* storage blocked: use default */ }
  sel.innerHTML = VOICES.map((v) => `<option value="${esc(v.name)}">${esc(v.name.replace(/^Microsoft |^Google /, ""))}${femaleRank(v) < 999 ? " ♀" : ""}</option>`).join("")
    || "<option>default voice</option>";
  if (saved && VOICES.some((v) => v.name === saved)) sel.value = saved;
}
function applyVoice(u) {
  const v = VOICES.find((x) => x.name === $("voice-select").value) || VOICES[0];
  if (v) { u.voice = v; u.lang = v.lang; } else u.lang = "en-US";
  u.rate = 0.97; u.pitch = 1.05; // slightly slower and warmer than the default
}
if ("speechSynthesis" in window) { loadVoices(); speechSynthesis.onvoiceschanged = loadVoices; }
$("voice-select").onchange = (e) => { try { localStorage.setItem("dms_voice", e.target.value); } catch (err) { /* ignore */ } };
$("voice-test").onclick = () => {
  if (!("speechSynthesis" in window)) return;
  speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance("Please look ahead at the road. Keep your eyes on the road.");
  applyVoice(u);
  speechSynthesis.speak(u);
};
$("sens-select").onchange = (e) => send({ type: "config", sensitivity: e.target.value });

function fmtTime(t) { const m = Math.floor(t / 60), s = Math.floor(t % 60); return `${m}:${String(s).padStart(2, "0")}`; }

// ---------------------------------------------------------------- chart
function drawChart(cv, hist, windowS) {
  const dpr = window.devicePixelRatio || 1, W = cv.clientWidth, H = cv.clientHeight;
  if (!W) return;
  cv.width = W * dpr; cv.height = H * dpr;
  const g = cv.getContext("2d"); g.scale(dpr, dpr);
  const pad = { l: 30, r: 8, t: 8, b: 18 }, w = W - pad.l - pad.r, h = H - pad.t - pad.b;
  const tMax = hist.length ? hist[hist.length - 1].t : 0, tMin = windowS ? tMax - windowS : (hist.length ? hist[0].t : 0);
  const X = (t) => pad.l + ((t - tMin) / Math.max(tMax - tMin, 1e-6)) * w, Y = (v) => pad.t + h - (v / 100) * h;
  // level bands
  [[30, 55, "rgba(227,179,65,.06)"], [55, 80, "rgba(240,136,62,.08)"], [80, 100, "rgba(248,81,73,.10)"]].forEach(([a, b, c]) => { g.fillStyle = c; g.fillRect(pad.l, Y(b), w, Y(a) - Y(b)); });
  g.strokeStyle = "#243042"; g.lineWidth = 1; g.fillStyle = "#8b98a9"; g.font = "10px system-ui";
  for (const v of [0, 30, 55, 80, 100]) { g.beginPath(); g.moveTo(pad.l, Y(v)); g.lineTo(W - pad.r, Y(v)); g.stroke(); g.fillText(v, 4, Y(v) + 3); }
  const span = tMax - tMin, step = span > 600 ? 120 : span > 120 ? 30 : 10;
  for (let t = Math.max(0, Math.ceil(tMin / step) * step); t <= tMax; t += step) g.fillText(fmtTime(Math.max(t, 0)), X(t) - 10, H - 4);
  const css = getComputedStyle(document.documentElement);
  for (const [k, c, lw] of [["d", "--c-d", 1.5], ["x", "--c-x", 1.5], ["a", "--c-a", 1.5], ["risk", "--c-risk", 2.2]]) {
    g.strokeStyle = css.getPropertyValue(c).trim(); g.lineWidth = lw; g.beginPath();
    hist.forEach((p, i) => (i ? g.lineTo(X(p.t), Y(p[k])) : g.moveTo(X(p.t), Y(p[k]))));
    g.stroke();
  }
}

// ---------------------------------------------------------------- report
function showReport(m) {
  S.report = m;
  const s = m.summary;
  const kpi = (k, v, sub = "") => `<div class="card stat"><div class="k">${k}</div><div class="v">${v}</div><div class="s">${sub}</div></div>`;
  const lv = s.time_in_level_pct;
  const kinds = Object.entries(s.alerts_by_kind).map(([k, v]) => `<tr><td>${k.replace(/_/g, " ")}</td><td>${v}</td></tr>`).join("") || `<tr><td class="muted">none</td><td></td></tr>`;
  const emos = Object.entries(s.emotion_time_pct).map(([k, v]) => `${k} ${v}%`).join(" · ") || "–";
  const scoreCls = s.safety_score >= 80 ? "good" : s.safety_score >= 60 ? "warn" : "bad";
  $("report-body").innerHTML = `
    <p class="muted">Session started ${s.started} · ${fmtTime(s.duration_s)} analysed · face visible ${s.face_visible_pct}% of frames</p>
    <div class="kpis">
      ${kpi("Safety score", `<span class="${scoreCls}">${s.safety_score}</span>`, "100 − average risk")}
      ${kpi("Alerts", s.alerts_total)}
      ${kpi("Peak risk", s.max_risk)}
      ${kpi("Peak PERCLOS", s.max_perclos_pct + "%")}
      ${kpi("Longest eye closure", s.longest_eye_closure_s + " s")}
      ${kpi("Longest look away", s.longest_eyes_off_road_s + " s")}
      ${kpi("Yawns", s.yawns)}
      ${kpi("Blinks / min", s.mean_blink_rate)}
    </div>
    <div class="muted small">Time in each risk level</div>
    <div class="levelbar">${Object.entries(lv).map(([k, v]) => `<div title="${k} ${v}%" style="width:${v}%;background:${LVL_COLOR[k]}"></div>`).join("")}</div>
    <div class="small">${Object.entries(lv).map(([k, v]) => `<span style="color:${LVL_COLOR[k]}">■</span> ${k} ${v}%`).join(" &nbsp; ")}</div>
    <div class="cols" style="margin-top:14px">
      <div><h3>Alerts by type</h3><table>${kinds}</table></div>
      <div><h3>Emotions</h3><p>${emos}</p><h3>Unsafe behaviour</h3><p>${Object.entries(s.behaviour_time_s).map(([k, v]) => `${k}: ${v}s`).join(" · ") || "none detected"}</p></div>
    </div>
    ${S.features && S.features.rating ? `<div class="kss-row">How sleepy did you feel? <select id="kss-select"><option value="">choose</option>
      ${["1 extremely alert", "2", "3 alert", "4", "5 neither", "6", "7 sleepy, no effort to stay awake", "8", "9 fighting sleep"]
        .map((x, i) => `<option value="${i + 1}">${x}</option>`).join("")}</select> <span class="muted small" id="kss-done"></span></div>` : ""}
    <h3>Risk over the whole session</h3>`;
  const kss = $("kss-select");
  if (kss) kss.onchange = () => { if (kss.value) { send({ type: "kss", value: Number(kss.value) }); $("kss-done").textContent = "saved, thank you"; } };
  $("report").classList.remove("hidden");
  const hist = m.timeline.map((p) => ({ t: p.t, risk: p.risk, d: p.drowsiness * 100, x: p.distraction * 100, a: p.affect * 100 }));
  requestAnimationFrame(() => drawChart($("report-chart"), hist, 0));
}
function download(name, text, type) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type })); a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

// ---------------------------------------------------------------- start / stop
async function start() {
  if (!S.ready) return alert("Still loading models – try again in a moment.");
  ensureAudio();
  S.hist = []; S.lastT = -1; S.inflight = false;
  $("placeholder").classList.add("hidden");
  if (S.src === "webcam") {
    try { S.stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false }); }
    catch (e) { return alert("Camera not available: " + e.message + (location.protocol === "http:" && location.hostname !== "localhost" ? "\n(Browsers only allow the camera on https or localhost.)" : "")); }
    video.srcObject = S.stream; video.classList.remove("hidden"); rawCanvas.classList.add("hidden");
    await video.play();
    S.t0 = performance.now();
    send({ type: "recalibrate" });
  } else if (S.src === "file") {
    if (!S.fileURL) { $("file-input").click(); return; }
    video.srcObject = null; video.src = S.fileURL; video.classList.remove("hidden"); rawCanvas.classList.add("hidden");
    video.controls = true; video.currentTime = 0;
    send({ type: "recalibrate" });
    S.running = true;
    await video.play();
  } else {
    const url = $("stream-url").value.trim();
    if (!url) return alert("Enter the camera stream URL (RTSP / HTTP MJPEG).");
    video.classList.add("hidden"); rawCanvas.classList.remove("hidden");
    send({ type: "stream_start", url });
  }
  S.running = true;
  $("start").textContent = "Stop"; $("start").className = "btn danger";
  startLoop();
  startAudio();
}
function stop(showRep) {
  S.running = false;
  if (S.stream) { S.stream.getTracks().forEach((t) => t.stop()); S.stream = null; }
  if (S.src === "file") video.pause();
  if (S.src === "stream") send({ type: "stream_stop" });
  if ("speechSynthesis" in window) speechSynthesis.cancel();
  stopAudio();
  $("sos").classList.add("hidden");
  $("start").textContent = "Start"; $("start").className = "btn primary";
  if (showRep) send({ type: "report" });
}

// ---------------------------------------------------------------- wiring
$("start").onclick = () => (S.running ? stop(true) : start());
$("recal").onclick = () => { send({ type: "recalibrate" }); S.hist = []; };
$("report-btn").onclick = () => send({ type: "report" });
$("print-btn").onclick = () => window.print();
$("report-close").onclick = () => $("report").classList.add("hidden");
$("dl-json").onclick = () => S.report && download("trip_report.json", JSON.stringify(S.report, null, 2), "application/json");
$("dl-csv").onclick = () => {
  if (!S.report || !S.report.timeline.length) return;
  const rows = S.report.timeline, keys = Object.keys(rows[0]);
  download("trip_timeline.csv", [keys.join(","), ...rows.map((r) => keys.map((k) => r[k] ?? "").join(","))].join("\n"), "text/csv");
};
$("opt-objects").onchange = (e) => send({ type: "config", objects: e.target.checked });
document.querySelectorAll("#source-seg button").forEach((b) => (b.onclick = () => {
  if (S.running) stop(false);
  document.querySelectorAll("#source-seg button").forEach((x) => x.classList.toggle("active", x === b));
  S.src = b.dataset.src;
  $("stream-url").classList.toggle("hidden", S.src !== "stream");
  $("src-label").textContent = { webcam: "webcam", file: "video file", stream: "network stream" }[S.src];
  video.controls = S.src === "file";
  if (S.src === "file") $("file-input").click();
}));
$("file-input").onchange = (e) => {
  const f = e.target.files[0]; if (!f) return;
  if (S.fileURL) URL.revokeObjectURL(S.fileURL);
  S.fileURL = URL.createObjectURL(f);
  $("src-label").textContent = f.name;
  video.srcObject = null; video.src = S.fileURL; video.controls = true;
  $("placeholder").classList.add("hidden");
};
video.addEventListener("ended", () => { if (S.running && S.src === "file") stop(true); });
video.addEventListener("play", () => { if (S.src === "file" && !S.running && S.fileURL && S.ready) { S.running = true; $("start").textContent = "Stop"; $("start").className = "btn danger"; ensureAudio(); send({ type: "recalibrate" }); startLoop(); startAudio(); } });
document.querySelectorAll(".tab").forEach((b) => (b.onclick = () => {
  document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === b));
  document.querySelectorAll(".tabpane").forEach((p) => p.classList.toggle("active", p.id === "tab-" + b.dataset.tab));
  if (b.dataset.tab === "about") loadInfo();
}));
window.addEventListener("resize", () => drawChart($("chart"), S.hist, 60));

function fillModels(models) {
  const rows = [["Face landmarks", models.landmarks], ["Eye state", models.eye], ["Emotion", models.emotion], ["Phone / objects", models.objects],
    ["Behaviour classifier", models.cabin], ["Temporal drowsiness", models.temporal], ["Frame drowsiness", models.drowsy_frame]];
  $("models-table").innerHTML = "<tr><th>Component</th><th>Model</th></tr>" + rows.map(([k, v]) =>
    `<tr><td>${k}</td><td>${v ? esc(v) : '<span class="muted">not loaded (train it, then pass the checkpoint)</span>'}</td></tr>`).join("");
}
async function loadInfo() {
  try {
    const info = await (await fetch("/api/info")).json();
    if (info.models) fillModels(info.models);
    if (info.metrics && info.metrics.length) {
      $("metrics-table").innerHTML = "<tr><th>Task</th><th>Accuracy</th><th>Macro-F1</th><th>Protocol</th></tr>" + info.metrics.map((m) =>
        `<tr><td>${esc(m.task)}<br><span class="muted small">${esc(m.run)}</span></td><td>${(m.accuracy * 100).toFixed(1)}%</td><td>${(m.macro_f1 * 100).toFixed(1)}%${m.std !== undefined ? ` ± ${(m.std * 100).toFixed(1)}` : ""}</td><td class="small">${esc(m.protocol)}</td></tr>`).join("");
    }
  } catch (e) { /* offline is fine */ }
}

updateStats({ features: {}, face: false, eye_openness: null, perclos: 0, blink_rate: null, yawns: 0, calibrating: null, behaviours: {}, detections: [], risk: null, emotion_probs: null });
connect();
