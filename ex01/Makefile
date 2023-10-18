UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
	CC=gcc
endif
ifeq ($(UNAME_S),Darwin)
	CC=gcc-12
endif
MPICC?=mpicc

perf?=0
ifeq "$(perf)" "1"
	CPPFLAGS += -D_PERF_
endif

CFLAGS+=-Wall -O2 -lm 
CFLAGS_THREADS=$(CFLAGS) -fopenmp
CFLAGS_THREADS_SIMD=$(CFLAGS_THREADS) -mavx2 -mfma -mno-recip -fno-math-errno -fno-signaling-nans -fexcess-precision=fast -march=core-avx2

all: diffusion2d_mpi_nb diffusion2d_mpi_nb_2 diffusion2d_mpi_nb_2_hybrid

diffusion2d_mpi_nb: diffusion2d_mpi_nb.c
	$(MPICC) $(CFLAGS) -o $@ $<

diffusion2d_mpi_nb_2: diffusion2d_mpi_nb_2.c
	$(MPICC) $(CFLAGS_THREADS_SIMD) -o $@ $<

diffusion2d_mpi_nb_2_hybrid: diffusion2d_mpi_nb_2_hybrid.c
	$(MPICC) $(CFLAGS_THREADS_SIMD) -o $@ $<

clean:
	rm -rf diffusion2d_mpi_nb diffusion2d_mpi_nb_2 diffusion2d_mpi_nb_2_hybrid
	rm -rf *.dat *.txt *.bin *.png

