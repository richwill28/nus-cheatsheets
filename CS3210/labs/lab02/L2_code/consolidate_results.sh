#!/bin/bash

for (( i = 1; i <= 256; i = $(($i * 2)) ))
do
	cat latest_program_1024_"${i}"_log.out >> observation-"${1}".txt 
done
