
#works with only 1 dot
extension(){
    echo "$(basename ${@##*.})"
}
filename(){
    echo "$(basename ${@%.*})"
}
########################

print_status (){
	if [ $? -eq 0 ]; then
		echo "OK!"
	else
		echo "Failed!"
	fi
}



if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: render_movie.sh [render_frames_folder] [output_name]"
else
    ### Only works if the files are .png though (even when we're not using an extension)
    echo -n "Rendering [ ${1}/%010d.png > $2.mp4 ] ... "
#    yes | ffmpeg  -r 30 -i ${1}/%010d.png $2.mp4  &>.ffmpeg_rendering.log
    yes | avconv -i ${1}/%010d.png $2.mp4  &>.ffmpeg_rendering.log
    if [ $? -eq 0 ]; then
        echo "OK!"
    else
        echo "Failed!"
        echo "Check the log: .ffmpeg_rendering.log"
    fi
fi
