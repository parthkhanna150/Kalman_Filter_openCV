#!/bin/bash

for file in ./*.mp4; do
	python ./ballestimator.py "$file"
done
