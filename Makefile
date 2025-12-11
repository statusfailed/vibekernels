CXX = g++
CXXFLAGS = -O3 -march=native -mavx2 -mavx512f -mavx512dq -mfma -std=c++17 -Wall -Wextra
LDFLAGS = -lopenblas -lgomp

SRCDIR = kernels
SOURCES = main.cpp $(wildcard $(SRCDIR)/*.cpp)
OBJECTS = $(SOURCES:.cpp=.o)
HEADERS = kernel_interface.hpp test_harness.hpp benchmark_harness.hpp tuning_harness.hpp
TARGET = run

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(OBJECTS) -o $@ $(LDFLAGS)

# Object files depend on their source and all headers
%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Main specifically depends on all headers since it includes them
main.o: main.cpp $(HEADERS)

clean:
	rm -f $(OBJECTS) $(TARGET)

# Convenience targets
test-reference: $(TARGET)
	./run test reference

bench-reference: $(TARGET)
	./run bench reference

list: $(TARGET)
	./run list

.PHONY: test-reference bench-reference list