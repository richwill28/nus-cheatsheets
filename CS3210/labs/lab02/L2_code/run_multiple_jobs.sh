#!/bin/bash

for (( i = 1; i <= 256; i = $(($i * 2)) ))
do
	./run_job.sh 1024 ${i} &
done
