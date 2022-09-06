# set base image (host OS)
FROM python:3.8.0-slim

# install some packages, for potential data fetching
RUN  apt-get update && apt-get install -y wget && apt-get install -y unzip

RUN useradd -m perona

USER perona

# set the working directory in the container
WORKDIR /home/perona/app

ENV PATH="/home/perona/.local/bin:${PATH}"
ENV PYTHONPATH "${PYTHONPATH}:/home/perona/app"

# install dependencies
COPY --chown=perona:perona requirements.txt .
RUN pip install --user -r requirements.txt

# Add everything else now
COPY --chown=perona:perona . .

# command to run on container start
CMD ping localhost