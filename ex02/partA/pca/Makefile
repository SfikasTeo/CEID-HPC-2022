CXX = g++
CPPFLAGS = -O3 -Wall -fopenmp
CPPFLAGS_SIMD += $(CPPFLAGS) -mavx2 -mfma
LIBS = -llapacke -lcblas -lopenblas
LDFLAGS = -lm -lpthread -lz -ldl

debug?=0
ifeq "$(debug)" "1"
	CPPFLAGS += -D_DEBUG_
endif

all: pca_threads
pca_threads: pca_threads.cpp
	@$(CXX) $(CPPFLAGS_SIMD) $(LIBS) $(LDFLAGS) -o $@ $<

clean:
	@rm -rf *.txt *.bin *.gz pca_threads 

