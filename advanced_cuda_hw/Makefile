NVCC=nvcc
TARGET=demo
SRC=test.cu

$(TARGET): $(SRC)
	$(NVCC) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET)
