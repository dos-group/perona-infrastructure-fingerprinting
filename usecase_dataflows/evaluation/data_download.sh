#!/bin/bash

PATH_SUFFIX="/home/perona/app"

if [ ! -d "$PATH_SUFFIX/data/scout_multiple" ]; then
  echo "Downloading dataset from public github repository..."
  wget -q --show-progress https://github.com/oxhead/scout/archive/refs/heads/master.zip -O $PATH_SUFFIX/data/scout_master.zip
  mkdir -p $PATH_SUFFIX/data/scout_multiple
  unzip -q $PATH_SUFFIX/data/scout_master.zip "scout-master/dataset/osr_multiple_nodes/*" -d $PATH_SUFFIX/data/tmp_extract
  mv $PATH_SUFFIX/data/tmp_extract/scout-master/dataset/osr_multiple_nodes/* $PATH_SUFFIX/data/scout_multiple
  rm -rf $PATH_SUFFIX/data/scout_master.zip $PATH_SUFFIX/data/tmp_extract
  echo "Finished preparing the source dataset."
fi