-- Driver Monitoring cloud schema (generated from dms/cloud/db.py; the backend also creates it on first start).
-- No column holds images, video or audio.

CREATE TABLE IF NOT EXISTS model_versions (
	id VARCHAR(32) NOT NULL, 
	kind VARCHAR(16) NOT NULL, 
	version INTEGER NOT NULL, 
	created_at FLOAT NOT NULL, 
	status VARCHAR(16) NOT NULL, 
	path TEXT NOT NULL, 
	sha256 VARCHAR(64) NOT NULL, 
	metrics JSON NOT NULL, 
	data JSON NOT NULL, 
	notes TEXT NOT NULL, 
	PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS ix_model_versions_kind ON model_versions (kind);
CREATE INDEX IF NOT EXISTS ix_model_versions_status ON model_versions (status);

CREATE TABLE IF NOT EXISTS users (
	id VARCHAR(32) NOT NULL, 
	email VARCHAR(254) NOT NULL, 
	name VARCHAR(120) NOT NULL, 
	password_hash VARCHAR(255) NOT NULL, 
	role VARCHAR(16) NOT NULL, 
	created_at FLOAT NOT NULL, 
	consent_training BOOLEAN NOT NULL, 
	consent_at FLOAT, 
	PRIMARY KEY (id)
);

CREATE UNIQUE INDEX IF NOT EXISTS ix_users_email ON users (email);

CREATE TABLE IF NOT EXISTS devices (
	id VARCHAR(32) NOT NULL, 
	user_id VARCHAR(32) NOT NULL, 
	name VARCHAR(120) NOT NULL, 
	kind VARCHAR(16) NOT NULL, 
	key_hash VARCHAR(64) NOT NULL, 
	key_prefix VARCHAR(12) NOT NULL, 
	created_at FLOAT NOT NULL, 
	last_seen FLOAT, 
	revoked BOOLEAN NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(user_id) REFERENCES users (id) ON DELETE CASCADE
);

CREATE UNIQUE INDEX IF NOT EXISTS ix_devices_key_hash ON devices (key_hash);
CREATE INDEX IF NOT EXISTS ix_devices_user_id ON devices (user_id);

CREATE TABLE IF NOT EXISTS events (
	id VARCHAR(64) NOT NULL, 
	user_id VARCHAR(32) NOT NULL, 
	device_id VARCHAR(32) NOT NULL, 
	trip_id VARCHAR(64) NOT NULL, 
	t_wall FLOAT NOT NULL, 
	t_session FLOAT, 
	kind VARCHAR(32) NOT NULL, 
	severity VARCHAR(16), 
	payload JSON NOT NULL, 
	feedback VARCHAR(16), 
	PRIMARY KEY (id), 
	FOREIGN KEY(user_id) REFERENCES users (id) ON DELETE CASCADE, 
	FOREIGN KEY(device_id) REFERENCES devices (id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS ix_events_device_id ON events (device_id);
CREATE INDEX IF NOT EXISTS ix_events_kind ON events (kind);
CREATE INDEX IF NOT EXISTS ix_events_t_wall ON events (t_wall);
CREATE INDEX IF NOT EXISTS ix_events_trip_id ON events (trip_id);
CREATE INDEX IF NOT EXISTS ix_events_user_id ON events (user_id);

CREATE TABLE IF NOT EXISTS samples (
	id VARCHAR(64) NOT NULL, 
	user_id VARCHAR(32) NOT NULL, 
	device_id VARCHAR(32) NOT NULL, 
	trip_id VARCHAR(64) NOT NULL, 
	created_at FLOAT NOT NULL, 
	label INTEGER NOT NULL, 
	label_source VARCHAR(16) NOT NULL, 
	data JSON NOT NULL, 
	quality JSON NOT NULL, 
	accepted BOOLEAN NOT NULL, 
	reject_reason TEXT, 
	PRIMARY KEY (id), 
	FOREIGN KEY(user_id) REFERENCES users (id) ON DELETE CASCADE, 
	FOREIGN KEY(device_id) REFERENCES devices (id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS ix_samples_accepted ON samples (accepted);
CREATE INDEX IF NOT EXISTS ix_samples_device_id ON samples (device_id);
CREATE INDEX IF NOT EXISTS ix_samples_trip_id ON samples (trip_id);
CREATE INDEX IF NOT EXISTS ix_samples_user_id ON samples (user_id);

CREATE TABLE IF NOT EXISTS trips (
	id VARCHAR(64) NOT NULL, 
	user_id VARCHAR(32) NOT NULL, 
	device_id VARCHAR(32) NOT NULL, 
	t_start FLOAT, 
	t_end FLOAT, 
	summary JSON NOT NULL, 
	kss INTEGER, 
	PRIMARY KEY (id), 
	FOREIGN KEY(user_id) REFERENCES users (id) ON DELETE CASCADE, 
	FOREIGN KEY(device_id) REFERENCES devices (id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS ix_trips_device_id ON trips (device_id);
CREATE INDEX IF NOT EXISTS ix_trips_user_id ON trips (user_id);

-- Supabase exposes tables in the public schema through its REST API. Our backend is the only client:
-- enable Row Level Security with NO policies, so the anon/authenticated API roles can read or write nothing.
-- (The backend connects as the database owner, which is not subject to RLS.)
ALTER TABLE model_versions ENABLE ROW LEVEL SECURITY;
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE devices ENABLE ROW LEVEL SECURITY;
ALTER TABLE events ENABLE ROW LEVEL SECURITY;
ALTER TABLE samples ENABLE ROW LEVEL SECURITY;
ALTER TABLE trips ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON ALL TABLES IN SCHEMA public FROM anon, authenticated;
