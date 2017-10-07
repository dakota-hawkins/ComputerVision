CC=gcc
CXX=g++
CPPFLAGS=$(shell pkg-config --cflags opencv)
LDLIBS=$(shell pkg-config --libs opencv)

SRCS=part2.cpp part2.hpp ../BinaryImageAnalysis/banalysis.cpp
OBJS=part2.out

$(OBJS): $(SRCS)
	$(CXX) -o $(OBJS) $(SRCS) $(CPPFLAGS) $(LDLIBS)
