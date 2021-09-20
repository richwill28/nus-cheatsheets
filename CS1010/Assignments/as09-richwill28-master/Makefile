.SILENT: test tidy
SHELL := /bin/bash
FILES=digits
CC=clang
LDLIBS=-lm -lcs1010
LIB_HOME=~cs1010/lib
CFLAGS=@compile_flags.txt -L $(LIB_HOME)

all: compile tidy

test:
	./test.sh digits 6 1
	./test.sh digits 100 30
	./test.sh digits 60000 30
	./test.sh digits 100 10000

all: compile tidy

compile: $(FILES) 

tidy: test
	clang-tidy -quiet *.c 2> /dev/null

clean:
	rm digits
# vim:noexpandtab
