set term pdfcairo font "Helvetica" enhanced  dashed #size 640,500
set output "Icg.pdf"


set grid


set style line 1 lc rgb '#0060ad' lt 2 lw 1.5 pt 7  ps 0.5
set pointintervalbox 1.6

set xlabel "f (Hz)"
set ylabel "a at 0º"

#set logscale x

set xrange [0:15001]

set xtics 1000
set mxtics 2

set xtics rotate by -45

plot "icg_table_1m" u 1:2 w linespoints ls 1  notitle

#pause -1