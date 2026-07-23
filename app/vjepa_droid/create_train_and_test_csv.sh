#!/bin/bash

DROID_PATH = /path/to/droid_raw

find $DROID_RAW_PATH | grep "trajectory.h5" | awk '{gsub(/\/trajectory.h5/, ""); print}' > dataset.csv

shuf dataset.csv > shuffled.csv

TOTAL=$(wc -l < shuffled.csv)
TRAIN_COUNT=$(( TOTAL * 90 / 100 ))

head -n $TRAIN_COUNT shuffled.csv > train.csv
tail -n +$(( TRAIN_COUNT + 1 )) shuffled.csv > test.csv

rm dataset.csv
rm shuffled.csv
