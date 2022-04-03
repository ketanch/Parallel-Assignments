#!/bin/sh

gcc -O3 -fopenmp omp_main_lock.c sync_library.c -o openmp_bench
gcc -O3 -pthread pthread_main_lock.c sync_library.c -o pthread_bench



echo "POSIX"
./pthread_bench 1
echo "-----------"
./pthread_bench 2
echo "-----------"
./pthread_bench 4
echo "-----------"
./pthread_bench 8
echo "-----------"
./pthread_bench 16

echo "OpenMP"
./openmp_bench 1
echo "-----------"
./openmp_bench 2
echo "-----------"
./openmp_bench 4
echo "-----------"
./openmp_bench 8
echo "-----------"
./openmp_bench 16