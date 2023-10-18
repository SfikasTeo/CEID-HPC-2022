DIM ?= 16
KNN ?= 32
TRA ?= 1048576
QUE ?= 1024
LOW ?=  0
HIGH ?= 2


CXX := gcc
NVXX := nvcc

CFLAGS  = -DPROBDIM=$(DIM) -DNNBS=$(KNN) -DTRAINELEMS=$(TRA) -DQUERYELEMS=$(QUE) -DLB=$(LOW) -DUB=$(HIGH)
CFLAGS += -DSURROGATES -DMANUAL
CFLAGS += -O3
LDFLAGS += -lm
NVFLAGS = --gpu-architecture=sm_35 -Wno-deprecated-gpu-targets 

all: knn_cuda gendata

gendata: gendata.c func.c
	$(CXX) $(CFLAGS) -o $@ $< $(LDFLAGS)

knn_cuda: knn_cuda.cu func.c
	$(NVXX) $(CFLAGS) $(NVFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f knn_cuda gendata *.txt
