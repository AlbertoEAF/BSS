#! /bin/bash


# Usage run_tests <dir>
dirtests="$1"

tests=$(find "$dirtests" -name '*.test')

log="run_tests.log"

rm -f "$log"

for t in $tests
do
    echo "$t"
    echo "Running $t..." >> "$log"
    time gen_tests.py "$t"
    if [[ $? -ne 0 ]]; then
	echo "LETHAL ERROR!" 
	echo "FAIL" >> "$log" 
	exit 1
    else
	echo "OK" >> "$log"
    fi
done
