# Makefile
SHELL:= /bin/bash

CXX?= g++
CXXFLAGS?= -O3 -Wall -std=c++11 -lstdc++

NVCC?= nvcc
NVFLAGS = -O3 --gpu-architecture=sm_35 -Wno-deprecated-gpu-targets

debug?=0
ifeq "$(debug)" "1"
	NVFLAGS += -DDEBUG
endif

all: diff2d_serial diff2d_cuda

diff2d_serial: diff2d_serial.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

diff2d_cuda: diff2d_cuda.cu
	$(NVCC) $(NVFLAGS) -o $@ $^

clean:
	rm -rf diff2d_serial diff2d_cuda *.txt *.dat
	
