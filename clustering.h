#ifndef CLUSTERING_HPP__
#define CLUSTERING_HPP__

#include "DUETstruct.h"
#include "Histogram2D.h"
#include "RankList.h"

void heuristic_pre_filter_clusters (Histogram<real> &hist, RankList<real,real> &preclusters, real min_peak_fall, real min_score)
{
  static const size_t skip_bins = 0; // skip the next bin if this one is below the noise threshold for faster performance (for sure a peak will not arise in the next bins)

  const size_t max_bin = hist.bins() - 1;	

  preclusters.clear();
  // Exclude bins on the border (Borderless histogram and interior region = interest region)
  for (size_t bin=1; bin < max_bin; ++bin)
    {
      real score = hist.bin(bin);

      if (score > min_score)
	{
	  if (score - hist.bin(bin-1) >= min_peak_fall && 
	      score - hist.bin(bin+1) >= min_peak_fall)
	    preclusters.add(score, hist.get_bin_center(bin));
	}
      else
	bin += skip_bins; // skip (faster but might not be safe - we don't want to skip over a peak) 
    }
}


void heuristic_pre_filter_clusters2D (Histogram2D<real> &hist, RankList<real, Point2D<real> > &preclusters, real min_peak_fall, real min_score)
{
  static const size_t skip_bins = 0; // skip the next bin if this one is below the noise threshold for faster performance (for sure a peak will not arise in the next bins)

  const size_t max_alpha_bin = hist.xbins() - 1;	
  const size_t max_delta_bin = hist.ybins() - 1;

  preclusters.clear();
  // Bins on the histogram border are filtered out! We only want peaks in the histogram's interior region (since we're using a borderless histogram we filter out peaks from outside the chosen interest region this way)
  for (size_t alphabin=1; alphabin < max_alpha_bin; ++alphabin)
    {
      for (size_t deltabin=1; deltabin < max_delta_bin; ++deltabin)
	{				
	  real score = hist.bin(alphabin,deltabin);

	  if (score > min_score)
	    {
	      if (score - hist.bin(alphabin-1,deltabin  ) >= min_peak_fall && 
		  score - hist.bin(alphabin+1,deltabin  ) >= min_peak_fall &&
		  score - hist.bin(alphabin  ,deltabin-1) >= min_peak_fall &&
		  score - hist.bin(alphabin  ,deltabin+1) >= min_peak_fall)
		preclusters.add(score, hist.get_bin_center(alphabin,deltabin));
	    }
	  else
	    deltabin += skip_bins; // skip the next one (faster) (if there's no points at this bin there isn't a peak in the next bin for sure) 
	}
    }
}

/*
/// Returns |a-b| for unsigned int type size_t
inline size_t subabs(size_t a, size_t b)
{
  return ( a > b ? a-b : b-a );
}
*/


// Checks if the distance between the 2 points is smaller than d_delta and d_alpha for each dimension respectively. Independent Box coordinates!
bool belongs (const Point2D<real> &a, const Point2D<real> &b, real max_distance_alpha, real max_distance_delta)
{
  return (std::abs(b.x-a.x) <= max_distance_alpha && std::abs(b.y-a.y) <= max_distance_delta ? 1 : 0);
}

/// Aggregates clusters to the biggest cluster inside a certain radius (in box coordinates)
void heuristic_aggregate_preclusters (RankList<real,real> &preclusters, const DUETcfg &DUET, real min_peak_distance)
{
  size_t size = preclusters.eff_size(DUET.noise_threshold);

  real max_score = preclusters.scores[0];
  for (size_t i=0; i < size; ++i)
    if (preclusters.scores[i] * DUET.max_peak_scale_disparity < max_score)
      {
	preclusters.del(i,DUET.noise_threshold);
	--size;
	--i; // just deleted, check again
      }
  if (DUET.aggregate_clusters) // if for test purposes: always ON at release.
    {
      for (size_t i=0; i < size; ++i)
	{
	  for (size_t cluster=0; cluster < i; ++cluster)
	    {
	      if (std::abs(preclusters.values[i]-preclusters.values[cluster]) <= min_peak_distance)
		{
		  preclusters.del(i,DUET.noise_threshold);
		  --size; // Just deleted an element. No need to process extra 0-score entries.
		  --i; // The rest of the list was pushed up, process the next entry which is in the same position.
		}
	    }
	}
    }
}

void heuristic_clustering(Histogram<real> &hist, RankList<real,real> &clusters, const DUETcfg &DUET, real min_peak_distance)
{
  heuristic_pre_filter_clusters(hist, clusters, 
				DUET.min_peak_fall, DUET.noise_threshold);
  //	cout << preclusters;
  heuristic_aggregate_preclusters(clusters, DUET, min_peak_distance);
}


/// Aggregates clusters to the biggest cluster inside a certain radius (in box coordinates)
void heuristic_aggregate_preclusters2D (RankList<real,Point2D<real> > &clusters, const DUETcfg &DUET)
{
  size_t size = clusters.eff_size(DUET.noise_threshold);

  real max_score = clusters.scores[0];

  for (size_t i=0; i < size; ++i)
    if (clusters.scores[i] * DUET.max_peak_scale_disparity < max_score)
      {
	clusters.del(i,DUET.noise_threshold);
	--size;
	--i; // just deleted, check again
      }
  if (DUET.aggregate_clusters) // if for test purposes: always ON at release.
    {
      for (size_t i=0; i < size; ++i)
	{
	  for (size_t cluster=0; cluster < i; ++cluster)
	    {
	      if (belongs(clusters.values[i], clusters.values[cluster], DUET.min_peak_dalpha, DUET.min_peak_ddelta))
		{
		  clusters.del(i,DUET.noise_threshold);
		  --size; // Just deleted an element. No need to process extra 0-score entries.
		  --i; // The rest of the list was pushed up, process the next entry which is in the same position.
		}
	    }
	}
    }
}

void heuristic_clustering2D(Histogram2D<real> &hist, RankList<real, Point2D<real> > &clusters, const DUETcfg &DUET)
{
  heuristic_pre_filter_clusters2D(hist, clusters, DUET.min_peak_fall, DUET.noise_threshold);
  //  cout << clusters;
  heuristic_aggregate_preclusters2D(clusters, DUET);
}


// L2-norm for a vector with start and end point a, b
real distance(const Point2D<real> &a, const Point2D<real> &b)
{
  return abs(b.x-a.x, b.y-a.y);
}

// Returns the index of the closest cluster in clusters.
size_t closest_cluster(const Point2D<real> &point, Buffer<Point2D<real> > &clusters)
{
  const size_t size = clusters.size();
	
  real dist, min_distance = FLT_MAX;
  size_t min_i = 0;

  // Find the closest cluster
  for (size_t i=0; i < size; ++i)
    {
      dist = distance(point, clusters[i]);
      if (dist < min_distance)
	{
	  min_distance = dist;
	  min_i = i;
	}
    }

  return min_i;
}


#endif // CLUSTERING.HPP
