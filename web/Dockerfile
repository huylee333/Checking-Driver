# Image for the Checking-Driver web service (FastAPI, CPU).
#
# What is inside : Python 3.11, the libraries of requirements-docker.txt (CPU torch), the web code.
# What is NOT inside: model weights, the demo trip (raw UAH-DriveSet files), any token.
#   The app downloads weights and demo from private Hugging Face repos at start-up when HF_TOKEN is set.
#
# Build : docker build -t checking-driver:dev web
# Run   : docker run --rm -p 7860:7860 --env-file web/.env.local checking-driver:dev
FROM python:3.11-slim

# Run as a normal user (uid 1000), not as root.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /home/user/app

# Layer 1: libraries. They change rarely, so they come first and Docker reuses this cached layer
# whenever only the code changes.
COPY --chown=user requirements.txt requirements-docker.txt ./
RUN pip install --user -r requirements-docker.txt

# Layer 2: the code. It changes often, so it comes last.
COPY --chown=user . .

# Empty folders owned by the user: the app downloads weights/demo into them at start-up.
RUN mkdir -p weights demo

EXPOSE 7860

# app.py downloads the weights while it is being imported, so the web server only starts listening
# after that finishes. The long start-period keeps Docker from marking the container unhealthy meanwhile.
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/', timeout=4)"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
