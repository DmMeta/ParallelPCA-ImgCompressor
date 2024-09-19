CXX = g++
CPPFLAGS = -O3 -Wall -fopenmp -std=c++23 -Wno-deprecated-enum-enum-conversion
CPPFLAGS_SIMD += $(CPPFLAGS) -mavx2 -mfma 
HEADERS = -I/opt/OpenBLAS/include -I/opt/ImgCompression/src -I/opt/opencv/include/opencv4/
OPENBLAS_LPATH = -L/opt/OpenBLAS/lib -L/opt/opencv/lib
LIBS = -lopenblas -lm -lpthread -lz -ldl -lopencv_core -lopencv_imgcodecs
LDFLAGS = -Wl,-rpath=/opt/OpenBLAS/lib:/opt/opencv/lib

debug?=0
ifeq "$(debug)" "1"
	CPPFLAGS_SIMD += -D_DEBUG_
endif
	
all: main
.PHONY: clean

main: pca.o main.o
	$(CXX) $(CPPFLAGS_SIMD) -o $@ $^  $(OPENBLAS_LPATH) $(LIBS) $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CPPFLAGS_SIMD) $(HEADERS) -c -o $@ $^

clean:
	@rm -rf *.txt *.o main 
