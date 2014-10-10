#set term png truecolor
#set output "degeneracies.png"

#set term postscript landscape enhanced color dashed "Helvetica" 14 
set term pdfcairo 
set output "degeneracies.pdf"

set xlabel "HPF cutoff frequency"
set ylabel "Degeneracy Count"

set grid y

set key left top

# w = bar width ( 3 columns => 1/3 > 0.3 )
w = 5

N=48

set boxwidth w
set style fill transparent solid 0.5 noborder

# Cut the bottom half of the plot if no negative degeneracies arise.

stats 'Degeneracies' u 3
max_0_degeneracies = STATS_max
min_0_degeneracies = STATS_min # always set to the symmetric of the maximum anyways.
stats 'Degeneracies' u 4
max_1_degeneracies = STATS_min
min_1_degeneracies = STATS_min
stats 'Degeneracies' u 5
max_2_degeneracies = STATS_max
min_2_degeneracies = STATS_min


if (min_1_degeneracies == 0 && min_2_degeneracies == 0) set yrange [0:max_0_degeneracies*1.1]

set xtic rotate by -45
#set format x "%g Hz"
#set xtics 20

plot 'Degeneracies' u ($2-w):3 w boxes lc rgb"green" t '0 Degeneracies',\
     '' u ($2):4:xtic(1) w boxes lc rgb"yellow" t '1 Degeneracy',\
     '' u ($2+w):5 w boxes lc rgb"red" t '2 Degeneracies' \
     #, N notitle, -N notitle 

#pause -1