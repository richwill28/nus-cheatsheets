.SILENT: test tidy
SHELL := /bin/bash
FILES=peak sort inversion
CC=clang
LDLIBS=-lm -lcs1010
LIB_HOME=~cs1010/lib
CFLAGS=@compile_flags.txt -L $(LIB_HOME)

all: compile tidy

compile: $(FILES) 

test: $(FILES)
	for question in $(FILES); do ./test.sh $$question; done

tidy: test
	clang-tidy -quiet *.c 2> /dev/null

clean:
	rm -f $(FILES) 
# vim:noexpandtab
