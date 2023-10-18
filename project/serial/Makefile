DIM ?= 16
KNN ?= 32
TRA ?= 1048576
QUE ?= 1024
LOW ?=  0
HIGH ?= 2

CXX := gcc

CFLAGS  = -DPROBDIM=$(DIM) -DNNBS=$(KNN) -DTRAINELEMS=$(TRA) -DQUERYELEMS=$(QUE) -DLB=$(LOW) -DUB=$(HIGH)
CFLAGS += -DSURROGATES -DMANUAL
CFLAGS += -O3 -Wall
CFLAGS_SIMD = $(CFLAGS) -mavx2 -mfma
LDFLAGS += -lm
all: gendata knn_serial knn_serial_simd

gendata: gendata.c func.c
	$(CXX) $(CFLAGS) -o $@ $< $(LDFLAGS)

knn_serial: knn_serial.c func.c
	$(CXX) $(CFLAGS) -o $@ $< $(LDFLAGS) 

knn_serial_simd: knn_serial_simd.c func.c
	$(CXX) $(CFLAGS_SIMD) -o $@ $< $(LDFLAGS) 

clean:
	rm -rf knn_serial knn_serial_simd gendata *.txt
