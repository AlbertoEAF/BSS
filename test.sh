#! /bin/bash

clear

folder="$1"
sim="$2"
duet="$3"
log="$4"

( [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ] || [ -z "$4" ] ) && echo "No arguments!" && exit 1


echo "r.."
r "${folder}/${duet}" > "${folder}/${log}"
[ "$?" -gt 150 ] && exit 1

echo "mcli.."
static=`get_config_var ${folder}/${duet} DUET.static_rebuild`
[ "$?" -ne 0 ] && exit 1

if [ $static -eq 1 ]
then
    mcli bss_eval_static >> "${folder}/${log}"
else
    mcli bss_eval_dynamic >> "${folder}/${log}"
fi


