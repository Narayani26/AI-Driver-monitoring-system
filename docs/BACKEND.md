# Cloud backend: accounts, devices, trips and the retraining loop

The in-car app does all detection on the device and works with no internet. The **cloud backend** (`dms/cloud/`) is where things come together:
- users log in
- dash cams and phones are registered
- trips and alerts are stored
- drivers mark alerts right or wrong
- consented, quality-checked data is collected to retrain the models

```
 ┌──────────── in the car (offline-first) ────────────┐            ┌──────────────── cloud ─────────────────┐
 camera ─► DriverMonitor ─► alerts (voice/buzzer)                 │  FastAPI  /api/...                      │
                 │                                                │   users ─┬─< devices                    │
                 ├─► local SQLite (events, trips, samples*) ──Syncer──►      ├─< trips ─< events (+feedback) │
                 │       store-and-forward, HTTPS, device key     │          └─< samples* ─► quality gate   │
 driver: "was this alert right?" · "how sleepy were you?" (KSS)   │  dashboard (login, devices, trips)      │
                                                                  │  export_dataset ─► retrain ─► new model │
 * samples only with the driver's opt-in                          └─────────────────────────────────────────┘
```

## Run it on the Mac (SQLite, nothing to install)

The backend runs in its own terminal (terminal 1):
```bash
python -m dms.cloud.app
```

1. Open http://localhost:9000, choose **Create one** and register. The password needs at least 10 characters, with an upper-case letter and a digit.
2. Under **Devices**, add e.g. "MacBook" (type: Laptop / web). Copy the **device key**: it is shown only once.
3. Tick **Share anonymised driving measurements** if you want to give training data.

Start the car app in terminal 2, pointing it at the backend:
```bash
DMS_SYNC_TOKEN=<device key> python -m dms.web.server --face-ckpt runs/face/best.pt --temporal-ckpt runs_temporal2/hybrid_binary/deploy.pt --audio --mishap --sync-url http://localhost:9000/api/ingest --share-training-data
```

Drive (or sit at the webcam), then:
- In the alert list, rate alerts **✓ yes / ✗ no**.
- Open **Trip report** and pick **How sleepy did you feel?**

The car app uploads every 30 s when online. Refresh the dashboard: the trip, its alerts and your ratings appear there.

To make yourself an admin, which shows the training-data quality panel:
```bash
python -m dms.cloud.app --make-admin you@example.com
```

## Database: what is used and how it works

The backend talks to its database through **SQLAlchemy**, the standard Python database toolkit. The code describes the tables once (`dms/cloud/db.py`). SQLAlchemy turns every read and write into SQL for whichever database `DMS_DATABASE_URL` points to, so the **same code runs on SQLite and on PostgreSQL**. Nothing in the app changes; only that one setting does.

| Where | Database | How to choose it |
|---|---|---|
| Laptop demo | **SQLite**: a real SQL database stored in one file, `~/.dms/cloud.db`. No server to install. | default (nothing to set) |
| Your own server | **PostgreSQL 16** in Docker, next to the backend | `docker compose up -d` (below) |
| Managed cloud | **Supabase**, AWS RDS, Google Cloud SQL, Neon…: all of them *are* PostgreSQL | `DMS_DATABASE_URL=<their connection string>` |

The tables are created automatically on first start. The same 7 backend tests run on both SQLite and a real PostgreSQL 16 server: locally, and in GitHub CI as a separate `postgres` job.

### Option A: PostgreSQL with Docker Compose (one command)
Create the settings file:
```bash
cp .env.example .env
```
Open `.env` and replace both values with random ones:
```bash
openssl rand -hex 32
```
Start the database and the backend:
```bash
docker compose up -d
```
The dashboard is at http://localhost:9000. The database is only reachable by the backend container: no port is opened to the outside. Its data lives in the `pgdata` volume and survives restarts.

### Option B: Supabase (set up: project "ai-driver-monitoring", Mumbai)
The tables, indexes and security rules are already in the project (migration `db/migrations/0001_init.sql`). All 6 tables have Row Level Security with no public policies, so Supabase's public API can read nothing; only our backend reads and writes.

1. In Supabase, open the project **ai-driver-monitoring**. Go to **Project Settings → Database → Reset database password**, set a new password with letters and digits only (symbols break the connection string), and save it somewhere safe.
2. Click **Connect** at the top, choose **Session pooler**, and copy the URI:
   `postgresql://postgres.<project-ref>:[YOUR-PASSWORD]@aws-…pooler.supabase.com:5432/postgres`
3. In the project folder, create `.env` (it is git-ignored). Use the copied URI, but start it with `postgresql+psycopg://`, put your password in, and add `?sslmode=require` at the end:
   ```bash
   cat > .env <<'EOF'
   DMS_DATABASE_URL=postgresql+psycopg://postgres.<project-ref>:YOURPASSWORD@aws-…pooler.supabase.com:5432/postgres?sslmode=require
   EOF
   ```
   Then add a random signing secret:
   ```bash
   echo "DMS_CLOUD_SECRET=$(openssl rand -hex 32)" >> .env
   ```
4. Start the backend:
   ```bash
   set -a; source .env; set +a; python -m dms.cloud.app
   ```
   Open http://localhost:9000 and create your account. Then, in a second terminal, make yourself admin (use your own email):
   ```bash
   set -a; source .env; set +a; python -m dms.cloud.app --make-admin you@example.com
   ```
5. In Supabase's **Table Editor** your account now appears in `users`, and trips, alerts and ratings appear as you drive.

### What is stored

| Table | What | Notes |
|---|---|---|
| users | account, role (driver/admin), training consent and when it was given | password: scrypt + salt |
| devices | dash cam / phone / laptop of a user | only the SHA-256 of the device key is stored; keys can be revoked |
| trips | one drive: start/end, summary (safety score, alerts, PERCLOS…), driver's KSS rating | id = the in-car session id |
| events | each alert: time, type, severity, detail, **feedback** (right / false alarm) | upload is idempotent (resends don't duplicate) |
| samples | 30 s window of numeric face measurements + label | only with consent; quality-checked; rejection reason kept |

There is **no column for images, video or audio** anywhere: footage never leaves the car.

### Two databases, two jobs
- **In the car**, a small SQLite file (`~/.dms/events.db`) holds alerts, trips and samples while there is no internet. It works offline and survives power loss (WAL mode).
- **In the cloud**, PostgreSQL holds every user's data. The car's Syncer sends its local rows there when online, then marks them as sent.

## The retraining loop (how the models get better from real drives)

It runs automatically once a week (`deploy/dms-retrain.timer`), or by hand:
```bash
python -m dms.cloud.retrain --base features/rldd
```
Each run either:
- **skips**, because there isn't enough real data yet, or
- **registers a new model version** as *approved* or *rejected*, with its test numbers. You see these in the dashboard's Admin → Model registry.

Devices then update themselves; replace `<device key>` with the key from the dashboard:
```bash
DMS_SYNC_TOKEN=<device key> python -m dms.cloud.fetch_model --url https://<backend> --out weights/fleet
```
That downloads the approved model and verifies its SHA-256 before use.


1. **Labels come from the driver**, never from the model itself:
   - **✗ no** on a drowsiness alert → that window is labelled *alert*: a real false alarm, the most valuable data.
   - **✓ yes** → labelled *drowsy*, but only as a weak label, used for training and never for testing.
   - **KSS** sleepiness rating at the end of the trip: 1–4 alert, 5–6 low vigilant, 7–9 drowsy. These are the same 3 levels as UTA-RLDD.
2. **Quality gate** (`dms/cloud/quality.py`). A window becomes training data only if it shows *the driver, driving*, measured well:

   | Rule | Why |
   |---|---|
   | driver's face visible ≥ 90% of the window | no measuring an empty seat |
   | head near the calibrated pose ≥ 70% of the time | eye shape can't be judged when the head is turned or tilted |
   | the tracked face never jumped to another person | passenger leaning in, or a driver swap |
   | driver was calibrated, with a plausible open-eye value | a bad calibration would corrupt the labels |
   | vehicle moving (if the phone knows its speed) | parked = not driving |
   | ≥ 20 s at ≥ 8 frames/s, ≤ 10% missing values | enough signal |

   In the car, passengers are already ignored. The system follows the driver's face (the largest, and the one in the same place as before) and never a back-seat face.
3. **Export and retrain:**
   ```bash
   python -m dms.cloud.export_dataset features/fleet
   ```
   This writes the same `.npz` format as the UTA-RLDD features, **split by driver**: no person is in both train and test. Names and emails never leave the database; drivers become pseudonyms such as `drv_3fa1…`.
   ```bash
   python -m dms.training.train_temporal features/fleet/train --model hybrid --binary --folds 5 --out runs_fleet/hybrid_binary
   ```
   Then test the current model on real drivers it never saw:
   ```bash
   python -m dms.training.train_temporal features/rldd --model hybrid --binary --test-dir features/fleet/test --out runs_fleet/rldd_on_fleet
   ```
4. Ship the new checkpoint to devices only if it beats the current one on the held-out real drivers and the false-alarm rate does not go up.

## Security and privacy

- **Logins:**
  - Passwords are stored as scrypt hashes and need at least 10 characters with an upper-case letter and a digit.
  - After 5 failed logins per account or address within 15 minutes, further attempts are blocked.
  - A wrong email or wrong password gets the same error, so attackers can't find out which accounts exist.
- **Login tokens:** HMAC-signed and valid for 12 h. The signing secret comes from `DMS_CLOUD_SECRET`, or else a 0600 file in `~/.dms`.
- **Devices:** each one has its own revocable key, and only the key's hash is stored. One account's device can never write into another account's trips (tested).
- **Consent:** opt-in, and revocable. Withdrawing it **deletes** the samples already given. `DELETE /api/me` erases the account and all its data (right to erasure).
- **Headers:** a strict Content-Security-Policy (no inline scripts), no-store caching, nosniff and no framing; request size is capped.

## Mobile app later

The API is plain JSON over HTTPS with bearer tokens, so an Android/iOS app uses it unchanged:
- `/api/auth/login` for the driver
- a device key for the phone that acts as the in-car unit
- the same `/api/ingest`

The in-car logic (`dms/pipeline.py`, `dms/fusion/*`) is what gets ported to the phone. See [PRODUCTION.md](PRODUCTION.md).
