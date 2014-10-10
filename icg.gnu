set grid


set style line 1 lc rgb '#0060ad' lt 1 lw 1 pt 7 pi -1 ps 0.5
set pointintervalbox 1.6

set xlabel "f (Hz)"
set ylabel "a"

set logscale x

plot "icg_table_1m" u 1:2 w linespoints ls 1  notitle

pause -1