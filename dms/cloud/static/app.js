"use strict";
const $ = (id) => document.getElementById(id);
let token = null;
try { token = sessionStorage.getItem("dms_token"); } catch (e) { /* storage blocked: log in each time */ }
let registering = false;
let currentTrip = null;

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const when = (t) => (t ? new Date(t * 1000).toLocaleString() : "–");
const dur = (s) => (s == null ? "–" : s >= 3600 ? `${(s / 3600).toFixed(1)} h` : s >= 60 ? `${Math.round(s / 60)} min` : `${Math.round(s)} s`);

async function api(path, opts = {}) {
  const r = await fetch(path, {
    method: opts.method || "GET",
    headers: { "Content-Type": "application/json", ...(token ? { Authorization: "Bearer " + token } : {}) },
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  const data = await r.json().catch(() => ({}));
  if (r.status === 401 && token) { logout(); throw new Error("session expired"); }
  if (!r.ok) throw new Error(typeof data.detail === "string" ? data.detail : "request failed");
  return data;
}

function setToken(t) { token = t; try { t ? sessionStorage.setItem("dms_token", t) : sessionStorage.removeItem("dms_token"); } catch (e) { /* ignore */ } }
function logout() { setToken(null); show(false); }

function show(loggedIn) {
  $("auth").classList.toggle("hidden", loggedIn);
  $("app").classList.toggle("hidden", !loggedIn);
  $("logout").classList.toggle("hidden", !loggedIn);
  if (!loggedIn) $("who").textContent = "";
}

$("toggle-auth").onclick = (e) => {
  e.preventDefault();
  registering = !registering;
  $("auth-title").textContent = registering ? "Create account" : "Log in";
  $("auth-submit").textContent = registering ? "Create account" : "Log in";
  $("name-row").classList.toggle("hidden", !registering);
  $("toggle-auth").textContent = registering ? "I already have an account" : "Create one";
  $("f-pass").autocomplete = registering ? "new-password" : "current-password";
  $("auth-err").textContent = registering ? "Password: at least 10 characters with an upper-case letter and a digit." : "";
};

$("auth-form").onsubmit = async (e) => {
  e.preventDefault();
  $("auth-err").textContent = "";
  try {
    const body = { email: $("f-email").value, password: $("f-pass").value };
    if (registering) body.name = $("f-name").value;
    const r = await api(registering ? "/api/auth/register" : "/api/auth/login", { method: "POST", body });
    setToken(r.token);
    $("f-pass").value = "";
    await load();
  } catch (err) { $("auth-err").textContent = err.message; }
};
$("logout").onclick = logout;

async function load() {
  const me = await api("/api/me");
  show(true);
  $("who").textContent = `${me.name || me.email}${me.role === "admin" ? " · admin" : ""}`;
  $("consent").checked = me.consent_training;
  await Promise.all([loadStats(), loadDevices(), loadTrips()]);
  if (me.role === "admin") { $("admin-card").classList.remove("hidden"); await loadAdmin(); }
}

function bars(el, entries, cls = "") {
  const max = Math.max(1, ...entries.map(([, v]) => v));
  el.innerHTML = entries.length ? entries.map(([k, v]) => `<div class="hbar-row"><span>${esc(String(k).replace(/_/g, " "))}</span>
    <div class="hbar-track"><div class="hbar-fill ${cls}" data-w="${(100 * v / max).toFixed(1)}"></div></div><b>${v}</b></div>`).join("")
    : '<p class="muted small">nothing yet</p>';
  el.querySelectorAll("[data-w]").forEach((d) => { d.style.width = d.dataset.w + "%"; }); // CSP: no inline style attributes
}

function trend(el, trips) {
  const pts = trips.filter((t) => t.safety_score != null).slice(0, 30).reverse();
  if (!pts.length) { el.innerHTML = '<p class="muted small">no trips yet</p>'; return; }
  const W = 600, H = 140, pad = 22, bw = Math.max(4, (W - pad) / pts.length - 4);
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", "safety score per trip");
  for (const y of [0, 50, 100]) {
    const yy = H - 16 - (y / 100) * (H - 26);
    const l = document.createElementNS(ns, "line");
    Object.entries({ x1: pad, x2: W, y1: yy, y2: yy, class: "axis" }).forEach(([k, v]) => l.setAttribute(k, v));
    const tx = document.createElementNS(ns, "text");
    Object.entries({ x: 0, y: yy + 3, class: "axis-label" }).forEach(([k, v]) => tx.setAttribute(k, v));
    tx.textContent = y;
    svg.append(l, tx);
  }
  pts.forEach((t, i) => {
    const h = (t.safety_score / 100) * (H - 26);
    const r = document.createElementNS(ns, "rect");
    Object.entries({ x: pad + 4 + i * (bw + 4), y: H - 16 - h, width: bw, height: h, rx: 2,
      class: "trend-bar" + (t.safety_score < 60 ? " bad" : t.safety_score < 80 ? " low" : "") }).forEach(([k, v]) => r.setAttribute(k, v));
    const tip = document.createElementNS(ns, "title");
    tip.textContent = `${when(t.t_end)}: ${t.safety_score}`;
    r.append(tip);
    svg.append(r);
  });
  el.classList.add("trend");
  el.replaceChildren(svg);
}

async function loadAdmin() {
  const [o, ds, ms] = await Promise.all([api("/api/admin/overview"), api("/api/admin/dataset"), api("/api/admin/models")]);
  $("admin-stats").innerHTML = [["Users", o.users], ["Devices", o.devices], ["Trips", o.trips], ["Alerts", o.alerts],
    ["Rated by drivers", o.rated], ["False alarms", o.false_alarms], ["Drivers giving data", ds.drivers]]
    .map(([k, v]) => `<div><b>${v ?? 0}</b><span class="muted small">${k}</span></div>`).join("");
  bars($("ds-labels"), Object.entries(ds.accepted_by_label));
  bars($("ds-fa"), Object.entries(ds.false_alarms_by_kind).sort((a, b) => b[1] - a[1]), "bad");
  const rej = Object.entries(ds.rejected_by_reason);
  $("ds-rejects").innerHTML = rej.length ? "<tr><th>Reason</th><th>Windows</th></tr>" + rej.map(([k, v]) => `<tr><td>${esc(k)}</td><td>${v}</td></tr>`).join("")
    : '<tr><td class="muted">none rejected</td></tr>';
  const f = (m, k) => (m && m[k] != null ? (k === "false_alarm_rate" ? (100 * m[k]).toFixed(1) + "%" : m[k].toFixed(3)) : "–");
  $("models").innerHTML = ms.length ? "<tr><th>Version</th><th>Status</th><th>Macro-F1 new / old</th><th>False alarms new / old</th><th>Decision</th><th></th></tr>" +
    ms.map((m) => `<tr><td>${esc(m.kind)} v${m.version}<div class="muted small">${when(m.created_at)}</div></td><td class="status-${esc(m.status)}">${esc(m.status)}</td>
      <td>${f(m.metrics.candidate, "window_macro_f1")} / ${f(m.metrics.current, "window_macro_f1")}</td>
      <td>${f(m.metrics.candidate, "false_alarm_rate")} / ${f(m.metrics.current, "false_alarm_rate")}</td>
      <td class="small">${esc(m.metrics.decision || "")}</td>
      <td>${m.status === "approved" ? "" : `<button class="btn small" data-approve="${esc(m.id)}">${m.status === "retired" ? "Roll back to this" : "Approve"}</button>`}</td></tr>`).join("")
    : '<tr><td class="muted">No models yet. Run <code>python -m dms.cloud.retrain</code> once drivers have shared data.</td></tr>';
  document.querySelectorAll("[data-approve]").forEach((b) => (b.onclick = async () => {
    if (!confirm("Make this the model every device downloads?")) return;
    await api(`/api/admin/models/${b.dataset.approve}/status`, { method: "POST", body: { status: "approved" } });
    loadAdmin();
  }));
}

async function loadStats() {
  const s = await api("/api/stats");
  bars($("kinds"), Object.entries(s.alerts_by_kind).filter(([k]) => k !== "feedback").sort((a, b) => b[1] - a[1]));
  const alerts = Object.values(s.alerts_by_kind).reduce((a, b) => a + b, 0);
  const fa = s.feedback.false_alarm || 0;
  $("stats").innerHTML = [["Trips", s.trips], ["Alerts", alerts], ["Marked wrong", fa], ["Samples shared", s.training_samples_given]]
    .map(([k, v]) => `<div><b>${v}</b><span class="muted small">${k}</span></div>`).join("");
}

async function loadDevices() {
  const ds = await api("/api/devices");
  $("devices").innerHTML = ds.length
    ? "<tr><th>Name</th><th>Type</th><th>Key</th><th>Last upload</th><th></th></tr>" + ds.map((d) =>
      `<tr><td>${esc(d.name)}</td><td>${esc(d.kind)}</td><td class="muted">${esc(d.key_prefix)}…</td><td>${when(d.last_seen)}</td>
       <td><button class="btn small" data-revoke="${esc(d.id)}">Remove</button></td></tr>`).join("")
    : '<tr><td class="muted">No devices yet. Add your dash cam or phone below.</td></tr>';
  document.querySelectorAll("[data-revoke]").forEach((b) => (b.onclick = async () => {
    if (!confirm("Remove this device? It will no longer be able to upload.")) return;
    await api("/api/devices/" + b.dataset.revoke, { method: "DELETE" });
    loadDevices();
  }));
}

$("dev-form").onsubmit = async (e) => {
  e.preventDefault();
  const r = await api("/api/devices", { method: "POST", body: { name: $("dev-name").value, kind: $("dev-kind").value } });
  $("dev-name").value = "";
  const box = $("new-key");
  box.classList.remove("hidden");
  box.textContent = `Device key for "${r.name}" (shown only once, copy it now):\n${r.device_key}\n\nStart the car app with:  --sync-url ${location.origin}/api/ingest --sync-token ${r.device_key}`;
  loadDevices();
};

async function loadTrips() {
  const ts = await api("/api/trips");
  trend($("trend"), ts);
  $("trips").innerHTML = ts.length
    ? "<tr><th>Ended</th><th>Duration</th><th>Safety score</th><th>Alerts</th><th>Sleepiness (KSS)</th></tr>" + ts.map((t) =>
      `<tr class="click" data-trip="${esc(t.id)}"><td>${when(t.t_end)}</td><td>${dur(t.duration_s)}</td><td>${t.safety_score ?? "–"}</td>
       <td>${t.alerts_total ?? 0}</td><td>${t.kss ?? "–"}</td></tr>`).join("")
    : '<tr><td class="muted">No trips uploaded yet.</td></tr>';
  document.querySelectorAll("[data-trip]").forEach((r) => (r.onclick = () => openTrip(r.dataset.trip)));
}

for (let i = 1; i <= 9; i++) $("kss").insertAdjacentHTML("beforeend", `<option value="${i}">${i}</option>`);
$("kss").onchange = async (e) => {
  if (!currentTrip || !e.target.value) return;
  await api(`/api/trips/${currentTrip}/kss`, { method: "POST", body: { kss: Number(e.target.value) } });
  loadTrips();
};

async function openTrip(id) {
  const t = await api("/api/trips/" + id);
  currentTrip = id;
  $("trip-card").classList.remove("hidden");
  $("trip-title").textContent = `Trip ended ${when(t.t_end)} · safety score ${t.summary.safety_score ?? "–"}`;
  $("kss").value = t.kss ?? "";
  $("events").innerHTML = t.events.length
    ? "<tr><th>Time</th><th>Alert</th><th>Detail</th><th>Was it right?</th></tr>" + t.events.map((e) =>
      `<tr><td>${e.t_session != null ? dur(e.t_session) : when(e.t_wall)}</td><td class="sev-${esc(e.severity)}">${esc(e.kind.replace(/_/g, " "))}</td>
       <td class="muted">${esc(e.detail)}</td>
       <td>${e.feedback ? esc(e.feedback.replace("_", " ")) : `<button class="btn small" data-fb="correct" data-ev="${esc(e.id)}">✓ right</button>
         <button class="btn small" data-fb="false_alarm" data-ev="${esc(e.id)}">✗ wrong</button>`}</td></tr>`).join("")
    : '<tr><td class="muted">No alerts on this trip.</td></tr>';
  document.querySelectorAll("[data-fb]").forEach((b) => (b.onclick = async () => {
    await api(`/api/events/${b.dataset.ev}/feedback`, { method: "POST", body: { feedback: b.dataset.fb } });
    openTrip(id); loadStats();
  }));
  $("trip-card").scrollIntoView({ behavior: "smooth" });
}

$("consent").onchange = async (e) => {
  if (!e.target.checked && !confirm("Stop sharing? Measurements you already shared will be deleted.")) { e.target.checked = true; return; }
  await api("/api/me", { method: "PATCH", body: { consent_training: e.target.checked } });
  loadStats();
};

if (token) load().catch(() => logout()); else show(false);
