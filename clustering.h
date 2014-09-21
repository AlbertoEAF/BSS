#ifndef CLUSTERING_HPP__
#define CLUSTERING_HPP__

#include <iostream>

#include "types.h"
#include "abs.h"
#include <stdio.h>
#include <float.h>

#include "DUETstruct.h"
#include "Histogram2DDeclaration.h"
#include "RankList.h"



void heuristic_pre_filter_clusters (Histogram<real> &hist, RankList<real,real> &preclusters, real min_peak_fall, real min_score);


void heuristic_pre_filter_clusters2D (Histogram2D<real> &hist, RankList<real, Point2D<real> > &preclusters, real min_peak_fall, real min_score);

/*
/// Returns |a-b| for unsigned int type size_t
inline size_t subabs(size_t a, size_t b)
{
  return ( a > b ? a-b : b-a );
}
*/


// Checks if the distance between the 2 points is smaller than d_delta and d_alpha for each dimension respectively. Independent Box coordinates!
bool belongs (const Point2D<real> &a, const Point2D<real> &b, real max_distance_alpha, real max_distance_delta);

/// Aggregates clusters to the biggest cluster inside a certain radius (in box coordinates)
void heuristic_aggregate_preclusters (RankList<real,real> &preclusters, const DUETcfg &DUET, real min_peak_distance);

void heuristic_clustering(Histogram<real> &hist, RankList<real,real> &clusters, const DUETcfg &DUET, real min_peak_distance);

/// Aggregates clusters to the biggest cluster inside a certain radius (in box coordinates)
void heuristic_aggregate_preclusters2D (RankList<real,Point2D<real> > &clusters, const DUETcfg &DUET);

void heuristic_clustering2D(Histogram2D<real> &hist, RankList<real, Point2D<real> > &clusters, const DUETcfg &DUET);


// L2-norm for a vector with start and end point a, b
real distance(const Point2D<real> &a, const Point2D<real> &b);

// Returns the index of the closest cluster in clusters.
size_t closest_cluster(const Point2D<real> &point, Buffer<Point2D<real> > &clusters);



#endif // CLUSTERING.HPP
