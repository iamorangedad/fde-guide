NVCC ?= nvcc
CFLAGS ?= -O2 -std=c++17

BINS := vector_add reduction matmul_tiled

.PHONY: all clean
all: $(BINS)

vector_add: vector_add.cu common.cuh
	$(NVCC) $(CFLAGS) -o $@ vector_add.cu

reduction: reduction.cu common.cuh
	$(NVCC) $(CFLAGS) -o $@ reduction.cu

matmul_tiled: matmul_tiled.cu common.cuh
	$(NVCC) $(CFLAGS) -o $@ matmul_tiled.cu

clean:
	rm -f $(BINS)
