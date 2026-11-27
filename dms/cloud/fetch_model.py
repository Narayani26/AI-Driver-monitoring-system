"""Device side of the model update: download the approved model and verify it.

    python -m dms.cloud.fetch_model --url https://<cloud> --out weights/fleet
    (device key from DMS_SYNC_TOKEN)  ->  then start the app with --temporal-ckpt weights/fleet/temporal.pt

The SHA-256 published by the server is checked before the file replaces the old one, so a
corrupted or tampered download is never used. The previous model is kept as *.prev.pt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from pathlib import Path


def _get(url: str, key: str):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    return urllib.request.urlopen(req, timeout=60)


def fetch(base: str, key: str, out: Path, kind: str = "temporal") -> dict:
    if not base.startswith("https://") and not base.startswith(("http://localhost", "http://127.0.0.1")):
        raise SystemExit("use https:// (plain http only for localhost)")
    with _get(f"{base.rstrip('/')}/api/models/latest?kind={kind}", key) as r:
        meta = json.load(r)
    out.mkdir(parents=True, exist_ok=True)
    dst, tmp = out / f"{kind}.pt", out / f"{kind}.download"
    info = out / f"{kind}.json"
    if info.exists() and json.loads(info.read_text()).get("sha256") == meta["sha256"] and dst.exists():
        print(f"already up to date: {kind} v{meta['version']}")
        return meta
    h = hashlib.sha256()
    with _get(f"{base.rstrip('/')}/api/models/{meta['id']}/download", key) as r, open(tmp, "wb") as f:
        for chunk in iter(lambda: r.read(1 << 20), b""):
            h.update(chunk)
            f.write(chunk)
    if h.hexdigest() != meta["sha256"]:
        tmp.unlink()
        raise SystemExit("checksum mismatch: download discarded, current model kept")
    if dst.exists():
        dst.replace(out / f"{kind}.prev.pt")
    tmp.replace(dst)
    info.write_text(json.dumps(meta, indent=2))
    print(f"installed {kind} v{meta['version']} -> {dst}  ({meta['metrics'].get('decision', '')})")
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True, help="cloud backend, e.g. https://dms.example.com")
    ap.add_argument("--key", default=os.environ.get("DMS_SYNC_TOKEN"), help="device key (default: env DMS_SYNC_TOKEN)")
    ap.add_argument("--out", type=Path, default=Path("weights/fleet"))
    a = ap.parse_args(argv)
    if not a.key:
        raise SystemExit("device key missing: --key or DMS_SYNC_TOKEN")
    fetch(a.url, a.key, a.out)


if __name__ == "__main__":
    main()
