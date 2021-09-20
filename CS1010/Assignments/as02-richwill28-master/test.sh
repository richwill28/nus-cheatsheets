#!/bin/bash
PROG=$1
if [ ! -e $PROG ]
then
	echo "$PROG does not exist.  Have you compiled it with make?"
	exit 1
fi

num_failed=0
i=1
while true
do
	if [ -e inputs/$PROG.$i.in ]
	then
		out=$(mktemp --suffix=$PROG)
		./$PROG < inputs/$PROG.$i.in > $out
		if [ `diff $out outputs/$PROG.$i.out | wc -l` -ne 0 ] 
		then
			echo -n "$PROG: incorrect output for test case $i: "
			cat inputs/$PROG.$i.in
			num_failed=$((num_failed + 1))
		fi
		rm $out
		i=$((i + 1))
	else
		break
	fi
done
if [ $num_failed -eq 0 ] 
then
	echo "$PROG: passed"
fi
