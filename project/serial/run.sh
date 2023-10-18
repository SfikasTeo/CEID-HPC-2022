#!/bin/bash

./gendata trainingset.txt queryset.txt
./knn_serial trainingset.txt queryset.txt
./knn_serial_simd trainingset.txt queryset.txt
