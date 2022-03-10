#!/bin/bash

BIN_NAME="sol2"
gcc -O3 -fopenmp -DLOCAL_TESTING solution_2.c -o $BIN_NAME

for dim_size in {4096,8192,16384,32768,45000}
do
    for threads in {12,}
    do
        average_time=0
        for cnt in {1..10}
        do
            time=`./${BIN_NAME} ${threads} ${dim_size} | tail -n1 | awk '{ print $2 }'`
            average_time=$((average_time+time))
            echo "n = ${dim_size}, thread = ${threads}, cnt = ${cnt}, time = ${time}"
        done
        average_time=$((average_time/10))
        echo "------n = ${dim_size}, thread = ${threads}, avg time = ${average_time}-------"
    done
done

