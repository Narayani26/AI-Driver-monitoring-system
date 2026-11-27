"""Cloud backend: accounts, devices (dash cams / phones), trips, alert events and
consented training data for continuous model improvement.

The in-car app (dms.web.server / dms.app.run) works fully offline and uploads
to this backend with its store-and-forward Syncer whenever a network is there.

    python -m dms.cloud.app                       # http://localhost:9000
    DMS_DATABASE_URL=postgresql+psycopg://... python -m dms.cloud.app --host 0.0.0.0
"""
