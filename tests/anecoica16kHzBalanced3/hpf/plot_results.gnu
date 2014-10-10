#plot "results_table" u 1:2 with lines, '' u 1:3 with lines, '' u 1:4 with lines, '' u 1:5 w lines

set grid
set key outside

set xlabel 'HPF cutoff (Hz)'
set ylabel '(dB)'

# http://www.gnuplotting.org/tag/linespoints/

#set style line 1 lc rgb '#0060ad' lt 1 lw 2 pt 7 pi -1 ps 1.5
#set style line 1 lc rgb '#0060ad' lt 2 lw 2 pt 7 pi -1 ps 1.5
#set style line 1 lc rgb '#0060ad' lt 3 lw 2 pt 7 pi -1 ps 1.5
#set style line 1 lc rgb '#0060ad' lt 4 lw 2 pt 7 pi -1 ps 1.5
#set pointintervalbox 3

# How to use the cool-looking lines
# plot 'plotting_data1.dat' with linespoints ls 1


plot for [col=3:9:2] 'Results' u 1:col w lines ls col-1 title columnheader , '' u 1:col:col+1 w yerrorbars notitle
#replot for [col=3:9:2] 'Iresults' u 1:col w lines ls col-1 title columnheader , '' u 1:col:col+1 w yerrorbars notitle







pause -1