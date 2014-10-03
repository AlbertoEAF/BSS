#! /bin/bash

[ $# != 3 ] && echo -e "\nUsage:\n\tdownsample <samplerate> <folder> <downsample_folder>\n" && exit 1

samplerate=$1
folder="$2"
folder_out="$3"


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
    
    sox "$file" -r "$samplerate" "$fileout"
done



