CC=gcc
CXX=g++
CPPFLAGS=$(shell pkg-config --cflags opencv)
LDLIBS=$(shell pkg-config --libs opencv)

SRCS=segmentation.cpp segmentation.hpp ../BinaryImageAnalysis/banlysis.cpp
OBJS=segmentation.out

$(OBJS): $(SRCS)
	$(CXX) -o $(OBJS) $(SRCS) $(CPPFLAGS) $(LDLIBS)
