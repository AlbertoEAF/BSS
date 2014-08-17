#! /bin/bash

### Plots

ybins=`cat h.cfg`

#plot_cmd_2d="plot \"%s\" using 1:2:3 with image, \"s.dat\" with points"
#plot_cmd_3d="splot \"%s\" with pm3d, \"s.dat\" with points"

plot_cmd_2d="plot  \"%s\" binary record=(${ybins},-1) format=\"%f\" u 1:2:3 w image, \"s.dat\" with points"
plot_cmd_3d="splot \"%s\" binary record=(${ybins},-1) format=\"%f\" u 1:2:3 w pm3d,  \"s.dat\" with points"
### set pm3d; splot "h.dat" binary record=(ybins,-1) format="%f" u 1:2:3 w pm3d


# 2D
rm -f hist_render2D/*
mkdir -p hist_render2D
gnuplot_render_frames.sh hist_dats hist_render2D render_cfg/hist2D.gnut "${plot_cmd_2d}" && render_movie.sh hist_render2D 2D


# 3D
rm -f hist_render3D/*
mkdir -p hist_render3D
gnuplot_render_frames.sh hist_dats hist_render3D render_cfg/hist3D.gnut "${plot_cmd_3d}" && render_movie.sh hist_render3D 3D

