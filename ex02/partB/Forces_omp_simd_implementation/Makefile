CXX = g++
CPPFLAGS = -O3 -Wall
CPPFLAGS_SIMD += $(CPPFLAGS) -mavx2 -mfma
CPPFLAGS_THREADS += $(CPPFLAGS_SIMD) -fopenmp
LDFLAGS = -lm

perf?=0
ifeq "$(perf)" "1"
	CPPFLAGS += -flto -march=native -ffast-math
endif

debug?=0
ifeq "$(debug)" "1"
	CPPFLAGS += -D_DEBUG_
endif

all: grav_forces_serial grav_forces_simd grav_forces_threads

grav_forces_serial: grav_forces_serial.cpp timer.hpp particles.h utils.h
	$(CXX) $(CPPFLAGS) $(LDFLAGS) -o $@ $<

grav_forces_simd: grav_forces_simd.cpp timer.hpp particles.h utils.h
	$(CXX) $(CPPFLAGS_SIMD) $(LDFLAGS) -o $@ $<

grav_forces_threads: grav_forces_threads.cpp timer.hpp particles.h utils.h 
	$(CXX) $(CPPFLAGS_THREADS) $(LDFLAGS) -o $@ $<

clean: 
	@rm -f grav_forces_serial grav_forces_simd grav_forces_threads 
	@rm -rf *.dat *.txt *.bin

