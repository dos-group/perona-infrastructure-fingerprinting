#!/bin/bash

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
VOLUME_MOUNTS="-v ${SCRIPTPATH}/data:/home/perona/app/data -v ${SCRIPTPATH}/artifacts:/home/perona/app/artifacts"
IMAGE_NAME="perona-container:dev"

echo "### Creating the necessary folders (if required)... ###"
mkdir -p $SCRIPTPATH/artifacts
mkdir -p $SCRIPTPATH/data

echo "### Building the docker image (if required)... ###"
docker build -t $IMAGE_NAME .

# download data, if required
echo "### Downloading data (if required)... ###"
docker run -it --rm $VOLUME_MOUNTS -t $IMAGE_NAME bash /home/perona/app/evaluation/data_download.sh

run_soo_experiment(){
  echo "### Run full experiments (this will take some time)... ###"
  PYTHON_CMD="python /home/perona/app/evaluation/eval_soo.py"
  docker run -it --rm $VOLUME_MOUNTS -t $IMAGE_NAME $PYTHON_CMD
}

analysis(){
  echo "### Create plots... ###"
  PYTHON_CMD="python /home/perona/app/evaluation/analysis_results.py"
  docker run -it --rm $VOLUME_MOUNTS -t $IMAGE_NAME $PYTHON_CMD
}

shell(){
  echo "### Open shell in container... ###"
  docker run -it --rm $VOLUME_MOUNTS -t $IMAGE_NAME bash
}

# Check if the function exists (bash specific)
if declare -f "$1" > /dev/null
then
  # call arguments verbatim
  "$@"
else
  # Show a helpful error
  echo "'$1' is not a known function name" >&2
  exit 1
fi