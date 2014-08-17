#! /bin/bash

data_folder="${1}"
output_folder="${2}"
cfgfile="${3}"
cfgfile_folder="render_cfg"

if [ -z "$1" ]; then 
    echo -e "\nGenMovie Usage:\n\t gen_movie \"data_folder\" \"output_folder\" \"cfgfile\"\n" 
    exit
fi

gnut_options=`cat ${cfgfile_folder}/${cfgfile}`

#works with only 1 dot
extension(){
    echo "$(basename ${@##*.})"
}
filename(){
    echo "$(basename ${@%.*})"
}
########################


# se n houver matches entao n passa a ser *.dat ele mesmo
#shopt -s nullglob 
for f in ${data_folder}/*.dat # shopt disabled -- if the folder is empty shit happens
do
    set_output="set output \"${output_folder}/$(filename ${f}).png\""
    plot_command="eval plot_cmd.' \"${f}\" '.plot_options"

    echo -e "$set_output\n $gnut_options\n $plot_command" | gnuplot
done
