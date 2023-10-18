DIM ?= 16
KNN ?= 16
TRA ?= 1048576
QUE ?= 1024
LOW ?=  0
HIGH ?= 2


CXX := gcc
NVXX := nvc++

CFLAGS  = -DPROBDIM=$(DIM) -DNNBS=$(KNN) -DTRAINELEMS=$(TRA) -DQUERYELEMS=$(QUE) -DLB=$(LOW) -DUB=$(HIGH)
CFLAGS += -DSURROGATES -DMANUAL
CFLAGS += -O3
LDFLAGS += -lm 
ACCFLAGS = -acc -Minfo=accel -ta=multicore,tesla

all: gendata knn_acc

gendata: gendata.c func.c
	$(CXX) $(CFLAGS) -o $@ $< $(LDFLAGS)

knn_acc: knn_acc.cpp func.c
	$(NVXX) $(CFLAGS) $(ACCFLAGS) -o $@ $< $(LDFLAGS) 

clean:
	rm -f gendata knn_acc *.txt
