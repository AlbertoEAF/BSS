#! /usr/bin/env bash

[ $# -ne 2 ] && echo "prgm <(j)oin/(s)plit> <folder>" && exit 1

mode="$1"
folder="$2"

function lpath () {
    echo "${1%/*}"
}
function rpath () {
    echo "${1##*/}"
}

function heal_path () {
    echo "${1//\/\//\/}"
}

FOLDER_SEP="_____"

cd "$folder"

if [ "$mode" == "j" ]; then # join mode #################################
    files=$( find -mindepth 2  -name '*.wav' )

    for file in $files
    do
	# Merge from the subfolders to the folder
	hierarchic_name="${file:2}"
	new_name="${hierarchic_name//\//$FOLDER_SEP}"

	if [ "$hierarchic_name" != "$new_name" ]; then
	    mv  "$hierarchic_name" "$new_name"
	fi

    done

else # split mode #########################################################
    files=$( ls | grep '.wav' ) # ls requires prepending the folder!!

    for file in $files
    do
	new_file="${file//${FOLDER_SEP}/\/}"
	mkdir -p `lpath "$new_file"`
	mv "$file" "$new_file"
    done
fi
