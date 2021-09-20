.SILENT: test tidy
SHELL := /bin/bash
FILES=contact social
CC=clang
LDLIBS=-lm -lcs1010
LIB_HOME=~cs1010/lib
CFLAGS=@compile_flags.txt -L $(LIB_HOME)

all: compile tidy

compile: $(FILES) life

test: $(FILES)
	for question in $(FILES); do ./test.sh $$question; done
	echo "life: not tested.  Read questions.md for instructions."

tidy: test
	clang-tidy -quiet *.c 2> /dev/null

clean:
	rm -f $(FILES) life
# vim:noexpandtab
