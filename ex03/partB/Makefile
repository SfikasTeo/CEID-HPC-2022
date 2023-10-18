CXX?= gcc
CFLAGS = -O3 -Wall -fopenmp
LIBS = -lm

NVXX?= nvc++
NVFLAGS = -O3 -Wall -fopenmp -acc -Minfo=all -ta=tesla

all: ants_serial ants_acc 

ants_serial: AntColonySerial.c
	$(CXX) $(CFLAGS) -o $@ $<

ants_acc: AntColonyOpenAcc.c 
	$(NVXX) $(NVFLAGS) -o $@ $<

clean:
	rm -rf *.txt ants_serial ants_acc