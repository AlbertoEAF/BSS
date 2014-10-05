#! /bin/bash

[[ $# -ne 1 ]] && echo "Usage: cleantests.sh <recursive_folder_search_folder>" && exit 1

ecologs=$(find "$1" -name '*.ecolog')
ecologis=$(find "$1" -name '*.ecologi')
bsslogs=$(find "$1" -name '*.bsslog')
bsslogis=$(find "$1" -name '*.bsslogi')

for f in $ecologs; do
    rm "$f"
done

for f in $ecologis; do
    rm "$f"
done

for f in $bsslogs; do
    rm "$f"
done

for f in $bsslogis; do
    rm "$f"
done
