#!/usr/bin/gnuplot
#
# Plotting the data of file plotting_data1.dat
#
# AUTHOR: Hagen Wierstorf

reset

# wxt
#set terminal wxt  size 350,262 enhanced font 'Verdana,10' persist
# png
#set terminal png size 500,300 enhanced font 'Verdana,10'
#set output 'T.png'
# svg
set terminal svg size 500,300 fname 'Verdana, Helvetica, Arial, sans-serif' \
fsize '10'
set output 'T2.svg'

# color definitions
set border linewidth 1.5
set style line 1 lc rgb '#0060ad' lt 1 lw 2 pt 7 ps 0.8 # --- blue
set style line 2 lc rgb '#dd181f' lt 1 lw 2 pt 5 ps 0.8 # --- red

set key top right # top right is optional

set xlabel "Nº cores"
set ylabel "t (s)"

set ytics 0.5
#set tics scale 5

#set xrange [0:5]
set yrange [1:5]

plot 'timesBetelgeuse.dat' index 2 with linespoints ls 1 title "Não-optimizado", \
     'timesBetelgeuse.dat' index 3 with linespoints ls 2 title "g++ -O2"
