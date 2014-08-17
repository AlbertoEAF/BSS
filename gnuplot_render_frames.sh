#! /bin/bash

if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ] || [ -z "${4}" ]; then
    echo "Usage: gnuplot_render_frames.sh [dat_folder] [render_folder] [gnut] [plot_cmd]"
    exit 1
fi

dat_folder=$1
render_folder=$2
gnut=$3
cmd="${4}"

gnut_options=`cat $gnut`

# Sets the internal gnuplot script  variables, runs the commands in gnut and then executes the render script
gnuplot -e "dat_folder='$dat_folder' ; render_folder='$render_folder'; cmd='${4}'"  $gnut  render_frames.gnuplot
