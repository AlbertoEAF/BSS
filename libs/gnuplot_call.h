#ifndef GNUPLOT_CALL_H
#define GNUPLOT_CALL_H

#ifdef _WIN32
#define __GNUPLOT_BIN "wgnuplot "
#else
#define __GNUPLOT_BIN "gnuplot "
#endif

#define GNUPLOT(cfg) __GNUPLOT_BIN#cfg

#endif
