DIM ?= 16
KNN ?= 32
TRA ?= 1048576
QUE ?= 1024
LOW ?=  0
HIGH ?= 2

CXX := gcc
MPICC := mpicc

CFLAGS  = -DPROBDIM=$(DIM) -DNNBS=$(KNN) -DTRAINELEMS=$(TRA) -DQUERYELEMS=$(QUE) -DLB=$(LOW) -DUB=$(HIGH)
CFLAGS += -DSURROGATES -DMANUAL -DDEBUG
CFLAGS += -O3 -Wall
CFLAGS_SIMD = $(CFLAGS) -mavx2 -mfma
LDFLAGS += -lm 
all: gendata knn_mpi knn_mpi_simd

gendata: gendata.c func.c
	$(CXX) $(CFLAGS) -o $@ $< $(LDFLAGS)

knn_mpi: knn_mpi.c func.c
	$(MPICC) $(CFLAGS) -o $@ $< $(LDFLAGS) 

knn_mpi_simd: knn_mpi_simd.c func.c
	$(MPICC) $(CFLAGS_SIMD) -o $@ $< $(LDFLAGS) 


clean:
	rm -rf knn_mpi knn_mpi_simd gendata *.txt
