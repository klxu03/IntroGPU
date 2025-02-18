# Compiler
NVCC = nvcc

# Compiler flags
NVCC_FLAGS = -O2

# Default target builds both executables
all: global_memory.exe host_memory.exe

# Build global_memory.exe from global_memory.cu
global_memory.exe: global_memory.cu
	$(NVCC) $(NVCC_FLAGS) -o global_memory.exe global_memory.cu

# Build host_memory.exe from host_memory.cu
host_memory.exe: host_memory.cu
	$(NVCC) $(NVCC_FLAGS) -o host_memory.exe host_memory.cu

# Clean up generated executables
clean:
	rm -f global_memory.exe host_memory.exe
