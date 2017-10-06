CC=gcc
CXX=g++
CPPFLAGS=$(shell pkg-config --cflags opencv)
LDLIBS=$(shell pkg-config --libs opencv)

SRCS=part1.cpp ../BinaryImageAnalysis/banalysis.hpp
OBJS=part1.out

# part1.cpp: ../BinaryImageAnalysis/banalysis.cpp

$(OBJS): $(SRCS)
	$(CXX) -o $(OBJS) $(SRCS) $(CPPFLAGS) $(LDLIBS)
