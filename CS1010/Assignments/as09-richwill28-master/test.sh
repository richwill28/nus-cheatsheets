#!/bin/bash
set -o nounset
set -o errexit

function control_c() {
  if [ -e $out ]
  then
    rm $out
  fi
}

trap control_c INT

PROG=$1
if [ ! -e $PROG ]
  then
    echo "$PROG does not exist.  Have you compiled it with make?"
    exit 1
  fi

readonly TRAINING_SAMPLES=$2
readonly TESTING_SAMPLES=$3
NUM_OF_TEST_CASES=1
num_failed=0
num_passed=0

out=$(mktemp --suffix=$PROG)
if [ "$TRAINING_SAMPLES" -eq 60000 ] || [ "$TESTING_SAMPLES" -eq 10000 ] 
  then
      TRAINING_DIR=~cs1010/as09
      TESTING_DIR=~cs1010/as09
  else
      TRAINING_DIR=inputs
      TESTING_DIR=inputs
fi

cat $TRAINING_DIR/train${TRAINING_SAMPLES}.in $TESTING_DIR/test${TESTING_SAMPLES}.in | ./$PROG > $out
status="$?"
if [ "$status" -ne "0" ]
  then
    num_failed=$((num_failed + 1))
    echo "$PROG: return non-zero status $status for test case $i"
  else if [ -e $out ]
    then
      if [ `diff -bB $out outputs/${TRAINING_SAMPLES}-${TESTING_SAMPLES}.out | wc -l` -ne 0 ]
        then
          echo "$PROG: training with $TRAINING_SAMPLES, testing with $TESTING_SAMPLES : incorrect output"
          num_failed=$((num_failed + 1))
        else
          num_passed=$((num_passed + 1))
      fi
      rm $out
    fi
  fi
if [ "$num_passed" -eq "$NUM_OF_TEST_CASES" ]
then
  echo "$PROG: training with $TRAINING_SAMPLES samples, testing with $TESTING_SAMPLES samples: passed"
fi
