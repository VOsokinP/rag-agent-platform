# Build context is the ingested checkout, e.g.:
#   docker build -f docker/runner.Dockerfile -t devagent-runner:fastapi data/repos/fastapi
FROM python:3.12-slim

RUN pip install --no-cache-dir uv \
 && useradd --uid 1000 --create-home runner

WORKDIR /src
# LICENSE is copied because the project declares license-files; the build
# backend refuses to produce metadata without it. uv.lock is what makes the
# baseline reproducible -- see below.
COPY pyproject.toml README.md LICENSE uv.lock ./
COPY fastapi ./fastapi

# Installed from the checkout's own lockfile, not resolved fresh. The project
# sets filterwarnings = ["error"], so a newer transitive dependency that merely
# deprecates something turns into a collection error: resolving freely put
# anyio 4.12 -> 4.15 and broke 444 test files before a patch was ever applied.
# A red baseline makes every answer about a patch a false positive.
#
# The project itself is installed too, but PYTHONPATH=/work shadows it at run
# time so the *patched* copy is what gets imported -- see the runner's comment.
RUN uv export --frozen --no-emit-project --no-hashes --no-default-groups \
        --all-extras --group tests -o /tmp/requirements.txt \
 && uv pip install --system -r /tmp/requirements.txt \
 && uv pip install --system --no-deps -e .

USER runner
WORKDIR /work
