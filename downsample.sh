#! /bin/bash

[ $# -lt 3 ] && echo -e "\nUsage:\n\tdownsample <samplerate> <folder> <downsample_folder> [channel] [lowervolumemultiplier_on_channel] [HPF]\n" && exit 1

samplerate=$1
folder="$2"
folder_out="$3"
channel="$4"
volume_multiplier="$5"
HPF="$6"

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

    # Set the HPF filter command part
    #hpf_cmd=""
    if [ $# -eq 6 ]; then
     	hpf_cmd="highpass $HPF"
    fi

    channels=$( soxi $file | grep Channels | awk '{ print $3 }' )

    # Set the remix command component part.
    remix_cmd=""
    if [ $# -eq 5 ] && [ "$channels" -eq "2" ]; then
	if [ $channel == 1 ]; then # Change the volume on channel 1.
	    remix_cmd="remix 1v${volume_multiplier} 2"
	else
	    remix_cmd="remix 1 2v${volume_multiplier}"
	fi
    fi
    
    

    sox "$file" -r "$samplerate" "$fileout" $remix_cmd $hpf_cmd


    echo -e "\n\n"
done



