#include "types.h"


template class Buffer<real>;
template class Buffer<int>;
template class Buffer<unsigned int>;
template class Buffer<long unsigned int>;

template std::ostream &operator << (std::ostream &, const Buffer<real> &buffer);
template std::ostream &operator << (std::ostream &, const Buffer<int> &buffer);
template std::ostream &operator << (std::ostream &, const Buffer<unsigned int> &buffer);
template std::ostream &operator << (std::ostream &, const Buffer<long unsigned int> &buffer);
