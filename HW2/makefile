CC=gcc
CXX=g++
CPPFLAGS=$(shell pkg-config --cflags opencv)
LDLIBS=$(shell pkg-config --libs opencv)

SRCS=Source.cpp
OBJS=Source.out

$(OBJS): $(SRCS)
	$(CXX) -o $(OBJS) $(SRCS) $(CPPFLAGS) $(LDLIBS)
