#!/bin/bash
function control_c() {
	if [ -e $out ]
		then
			rm -f $out
		fi
}

trap control_c INT

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
		status="$?"
		if [ "$status" -ne "0" ]
		then
			echo "$PROG: return non-zero status $status for test case $i"
			# cat inputs/$PROG.$i.in
			num_failed=$((num_failed + 1))
		else 
			if [ -e $out ] 
			then
				if [ `diff -bB $out outputs/$PROG.$i.out | wc -l` -ne 0 ] 
				then
					echo "$PROG: incorrect output for test case $i "
					#cat inputs/$PROG.$i.in
					num_failed=$((num_failed + 1))
				fi
				rm -f $out
			else
				echo "$PROG: cannot find output file. Execution interrupted?"
				num_failed=$((num_failed + 1))
			fi
		fi
		i=$((i + 1))
	else
		break
	fi
done

if [ $i -eq 1 ] 
then
	echo "$PROG: no test cases found"
elif [ $num_failed -eq 0 ] 
then
	echo "$PROG: passed"
fi
# vim:noexpandtab:sw=4:ts=4
