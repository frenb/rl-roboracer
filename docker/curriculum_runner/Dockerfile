# curriculum-runner: chains curriculum stages as jobs (rl_agent/curriculum).
# No GPU and no training code: it only reads and writes Mongo. The code is
# bind-mounted (./rl_agent -> /python_ws/src), so edits need only a restart.
FROM python:3.12-slim

RUN pip install --no-cache-dir "pymongo>=4.6,<5"

WORKDIR /python_ws/src
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
CMD ["python", "-m", "curriculum.service"]
