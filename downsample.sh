#! /bin/bash

[ $# < 3 ] && echo -e "\nUsage:\n\tdownsample <samplerate> <folder> <downsample_folder> [channel] [lowervolumemultiplier]\n" && exit 1

samplerate=$1
folder="$2"
folder_out="$3"
channel="$4"

files=$( find "$folder" -name '*.wav' )

for file in $files
do
    filedir="${file%/*}"
    filename="${file##*/}"
    destdir="$folder_out/${filedir:${#folder}}"
    destdir="${destdir//\/\//\/}"
    fileout="$destdir/$filename"
    fileout="${fileout//\/\//\/}"
    echo "$file->$fileout "

    mkdir -p "$destdir"
    
    if [ $# == 3 ]; then
	sox "$file" -r "$samplerate" "$fileout"
    else
	#[ $# != 5 ] && exit(1)
	if [ $channel == 1 ]; then
	    sox "$file" -r "$samplerate" "$fileout" remix "1v$5" 2
	else
	    sox "$file" -r "$samplerate" "$fileout" remix 1 "2v$5"
	fi
    fi

done



