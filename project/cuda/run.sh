#!/bin/bash

./gendata trainingset.txt queryset.txt
./knn_cuda trainingset.txt queryset.txt