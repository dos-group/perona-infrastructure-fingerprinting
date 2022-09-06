# set base image (host OS)
FROM python:3.8-slim

RUN useradd -m perona

USER perona

# set the working directory in the container
WORKDIR /home/perona/benchmarking-operator/backend

ENV PATH="/home/perona/.local/bin:${PATH}"
ENV PYTHONPATH "${PYTHONPATH}:/home/perona/benchmarking-operator/backend"

# install dependencies
COPY --chown=perona:perona requirements.txt .
RUN pip install --user -r requirements.txt

# Add everything else now
COPY --chown=perona:perona . .

ENTRYPOINT ["python", "main.py"]