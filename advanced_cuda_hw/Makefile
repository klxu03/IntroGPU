NVCC=nvcc
TARGET=exec
SRC=image_identification.cu
CFLAGS=-I/usr/local/cuda/include
LDFLAGS=-L/usr/local/cuda/lib64 -lcudnn -lcufft -lcublas
$(TARGET): $(SRC)
	$(NVCC) $(SRC) -o $(TARGET) $(CFLAGS) $(LDFLAGS)
clean:
	rm -f $(TARGET)
