# Edge / server image for the driver monitoring system (CPU, x86-64 or ARM64 e.g. Raspberry Pi 5, Jetson in CPU mode).
#   docker build -t dms .
#   # dashboard on this machine only (camera frames come from the browser):
#   docker run --rm -p 127.0.0.1:8000:8000 dms
#   # headless in-car mode reading a Wi-Fi dash cam, speaking alerts, keeping an offline log:
#   docker run --rm --device /dev/snd -v dms-data:/data dms \
#       python -m dms.app.run --source rtsp://192.168.1.254/live --headless --speak --mishap --store /data/events.db
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        libegl1 libgles2 libgl1 libglib2.0-0 espeak-ng curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# CPU-only PyTorch keeps the image small; the rest from requirements (opencv headless for servers)
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && sed 's/^opencv-python>=.*/opencv-python-headless>=4.9/' requirements.txt > /tmp/req.txt \
    && pip install -r /tmp/req.txt

COPY dms ./dms
COPY weights ./weights
COPY scripts ./scripts
RUN cd weights && sha256sum -c --ignore-missing SHA256SUMS

# never run as root: a compromised process can't touch the rest of the device
RUN useradd --create-home --uid 10001 dms && mkdir -p /data && chown dms /data
USER dms
VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD curl -fs http://127.0.0.1:8000/api/health || exit 1

# Inside the container the server must listen on 0.0.0.0, so an access token is required:
# pass -e DMS_TOKEN=<secret> (otherwise a random one is printed in the logs).
CMD ["python", "-m", "dms.web.server", "--host", "0.0.0.0", "--port", "8000", "--store", "/data/events.db", "--audio", "--mishap"]
