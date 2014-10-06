#!/bin/bash

#[[ $# -ne 2 ]] && echo "Usage: prgm <recursive_search_folder>" && exit 1


dirtests="$1"

tests=$(find "$dirtests" -name '*.test')


for t in $tests
do
    echo -e "\n\nParsing test $t"
    parse_test.py "$t"
    if [[ $? -ne 0 ]]; then
	echo "LETHAL ERROR!" 
	echo "FAIL" 
	exit 1
    fi
done
