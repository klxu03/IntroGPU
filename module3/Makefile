# Default targets: build both executables
all: assignment_c.exe assignment.exe assignment_time_partition.exe

# 1) Build the original C assignment
assignment_c.exe: assignment.c
	gcc assignment.c -o assignment_c.exe

# 2) Build the CUDA assignment
assignment.exe: assignment.cu
	nvcc assignment.cu -o assignment.exe

# 3) Build the time partition CUDA assignment
assignment_time_partition.exe: assignment_time_partition.cu
	nvcc assignment_time_partition.cu -o assignment_time_partition.exe

# Optional clean target
clean:
	rm -f assignment_c.exe assignment.exe assignment_time_partition.exe
