#! /bin/bash

[ -z "$1" ] && echo "Usage: prgm <testdir>" && exit 1

dirs=$( find "$1" -type d )

#works with only 1 dot
extension(){
    echo "$(basename ${@##*.})"
}
filename(){
    echo "$(basename ${@%.*})"
}

wd=`pwd`

for dir in  $dirs 
do
    echo "### Processing folder <$dir>"
    Dir="${wd}/${dir}"
    duet_files=`ls ${Dir}/*.duet 2>/dev/null`
    [ "$?" -ne 0 ] && continue
    asim_files=`ls ${Dir}/*.asim 2>/dev/null`
    [ "$?" -ne 0 ] && continue

    for asim_file in "$asim_files"
    do
	echo "Asim.."
	csim "${folder}/${asim_file}" 1>/dev/null
	[ "$?" -ne 0 ] && echo "Aborting" && exit 1

	for duet_file in "$duet_files"
	do
	    echo "$duet_file"

	    # sh test.sh
	    echo "$dir" "$asim_file" "$duet_file" "$(filename ${asim_file})__$(filename ${duet_file}).log"
	    sh test.sh "$dir" "$(basename ${asim_file})" "$(basename ${duet_file})" "$(filename ${asim_file})__$(filename ${duet_file}).log"
	done
    done

done
