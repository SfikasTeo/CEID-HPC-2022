#!/bin/bash

# 469 700 	elvis_c
# 496 700 	elvis_r
# 4096 4096 cyclone
# 9500 9500 earth

# execute script as ./run.sh {1,2,3,4 = chose image} 'c' { # = number of principal Components} { int = Number of Threads}

principalComponents=$3

# Export the environmental variables for constrolling the multithreading capabilities.
export OPENBLAS_NUM_THREADS=$4
export GOTO_NUM_THREADS=$4
export OMP_NUM_THREADS=$4

if [ $1 -eq 1 ]
then
	./pca_threads -m 469 -n 700 -npc $principalComponents -if 'elvis' -features $2 -threads $4
fi

if [ $1 -eq 2 ]
then
	./pca_threads -m 469 -n 700 -npc $principalComponents -if 'elvis_new_ij' -features $2 -threads $4
fi

if [ $1 -eq 3 ]
then
	./pca_threads -m 4096 -n 4096 -npc $principalComponents -if 'cyclone' -features $2 -threads $4
fi

if [ $1 -eq 4 ]
then
	./pca_threads -m 9500 -n 9500 -npc $principalComponents -if 'earth' -features $2 -threads $4
fi