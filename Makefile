CXX = g++
CXXFLAGS = -O3 -march=native -std=c++17 -Wall -Wextra
LDFLAGS = -lopenblas

SRCDIR = kernels
SOURCES = main.cpp $(wildcard $(SRCDIR)/*.cpp)
OBJECTS = $(SOURCES:.cpp=.o)
TARGET = run

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(OBJECTS) -o $@ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

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