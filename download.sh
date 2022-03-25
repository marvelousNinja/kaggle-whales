#!/bin/bash
set -e
set -v

mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle
cd ./data

kaggle competitions download -c happy-whale-and-dolphin
unzip happy-whale-and-dolphin.zip

kaggle datasets download -d phalanx/whale2-cropped-dataset
unzip whale2-cropped-dataset.zip
