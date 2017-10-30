CC=gcc
CXX=g++
CPPFLAGS=$(shell pkg-config --cflags opencv) -lboost_regex hungarian-algorithm-cpp-master/Hungarian.cpp
LDLIBS=$(shell pkg-config --libs opencv)

SRCS=$(shell echo *.cpp)
OBJS=$(SRCS:.cpp=.out)

$(OBJS): $(SRCS)
	$(CXX) -o $(OBJS) $(SRCS) $(CPPFLAGS) $(LDLIBS)
