// Ranks must always be aggregated otherwise the return values might be wrong. So make sure after you call aggregate after completing elements deletion 

#ifndef RANKLIST_H__
#define RANKLIST_H__

#include "BufferDeclaration.h"

template <class Tscore, class Tvalue> class RankList;
template <class Tscore, class Tvalue> std::ostream &operator << (std::ostream &, RankList<Tscore,Tvalue> &);

template <class Tscore, class Tvalue>
class RankList
{
	friend std::ostream &operator << <>(std::ostream &, RankList<Tscore,Tvalue> &);

public:
	RankList (size_t elems, Tscore default_score=0, Tvalue default_value=0); 
	RankList (RankList<Tscore,Tvalue> &copy);


	void copy(RankList<Tscore,Tvalue> &copy);

	int add(Tscore score, Tvalue value);

	void clear() { scores.clear(); values.clear(); }

	void del_(size_t i);// Call aggregate in the end of the del_ calls to guarantee consistency of the ranks
	void del (size_t i, Tscore min_value); 
	void aggregate(Tscore min_value);

	size_t size() { return _size; }
	size_t eff_size(Tscore min_value); // number of valid scores

	void print(size_t entries = 0);

	Buffer<Tscore> scores;
	Buffer<Tvalue> values;
private:
	size_t _size;
	Tscore _init_score;
	Tvalue _init_value;
};

template <class Tscore, class Tvalue>
RankList<Tscore,Tvalue>::RankList(size_t elems, Tscore default_score, Tvalue default_value) 
 : scores(elems,default_score), values(elems,default_value), _size(elems), _init_score(default_score), _init_value(default_value)
{
}

template <class Tscore, class Tvalue>
RankList<Tscore,Tvalue>::RankList(RankList<Tscore,Tvalue> &copy) 
  :scores(copy.scores), values(copy.values), _size(copy._size), _init_score(copy._init_score), _init_value(copy._init_value)
{
}


template <class Tscore, class Tvalue>
  void RankList<Tscore,Tvalue>::copy(RankList<Tscore,Tvalue> &copy)
{
  Assert(_size == copy._size, "Sizes must match.");
  _init_score = copy._init_score;
  _init_value = copy._init_value;

  scores.copy(copy.scores);
  values.copy(copy.values);
}


template <class Tscore, class Tvalue>
int RankList<Tscore,Tvalue>::add(Tscore score, Tvalue value)
{
	size_t rank = _size;
	while (rank && score > scores[rank-1])
		--rank;

	if (rank < _size)
	{
		// Shift down in block the worse scores and items
		for (int i=_size-1; i != rank; --i)
		{
			scores[i] = scores[i-1];
			values[i] = values[i-1];
		}

		scores[rank] = score;
		values[rank] = value;

		return rank;
	}
	else
		return -1; // Score not good enough to fit the RankedList
}

template <class Tscore, class Tvalue>
void RankList<Tscore,Tvalue>::del_(size_t i)
{
	scores[i] = _init_score;
	values[i] = _init_value;
}


template <class Tscore, class Tvalue>
void RankList<Tscore,Tvalue>::del(size_t i, Tscore min_value)
{
	del_(i);
	aggregate(min_value);
}

template <class Tscore, class Tvalue>
void RankList<Tscore,Tvalue>::aggregate(Tscore min_value)
{
	for (size_t i=0; i < _size-1; ++i)
	{
		if (scores[i] < min_value)
		{
			// Pull up lower scores in block
			for (size_t p=i; p < _size-1; ++p)
			{
				scores[p] = scores[p+1];
				values[p] = values[p+1];
			}
			del_(_size-1);
		}
	}
}

template <class Tscore, class Tvalue>
size_t RankList<Tscore,Tvalue>::eff_size(Tscore min_value)
{
	/* The last element is tested first
	because in many scenarios the ranking can be full so there's
	no need to search the whole list. Check the last value first.
	Then run the whole list except the last value. 
		(same number of tests as running the list in order 
		 but better performance when there is a non-0 probability
		 of having full rankings on large lists)
	*/
	if (scores[_size-1] >= min_value)
		return _size;

	// size-1 optimization: last element has been tested already.
	for (size_t i=0; i < _size-1; ++i) 
		if (scores[i] < min_value)
			return i;

	return _size-1; // In case only the last element is below the score
}


template <class Tscore, class Tvalue>
std::ostream &operator << (std::ostream &output, RankList<Tscore,Tvalue> &rlist)
{
  for (size_t i = 0; i < rlist._size ; ++i)
    output << i << ")\t" << rlist.scores[i] << "\t\t" << rlist.values[i] << std::endl;  
  output << std::endl;

  return output; // allows chaining
}


template <class Tscore, class Tvalue>
void RankList<Tscore,Tvalue>::print(size_t entries)
{
  size_t I = ( entries == 0 ? _size : std::min(entries,_size) );

  for (size_t i = 0; i < I ; ++i)
    std::cout << i << ")\t" << scores[i] << "\t\t" << values[i] << std::endl;  
  std::cout << std::endl;
}



#endif // RANKLIST_H__
