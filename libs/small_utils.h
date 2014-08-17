#ifndef SMALL_UTILS_H__
#define SMALL_UTILS_H__

// Doesn't matter if it is a pointer or a standard variable, just throw them!
template <class T> 
void swap (T &a, T &b)
{
  T tmp = a;
  a = b;
  b = tmp;
}

/*
// USE THE ADDRESS OF THE SWAPPABLE VARS ALWAYS!! EVEN IN POINTERS!!
template <class T> 
void swap (T * a, T * b)
{
  T tmp = *a;
  *a = *b;
  *b = tmp;
}
*/


/// Prompt a variable until a valid result is given.
template <class T>
T prompt (const std::string &prompt, T *var)
{
  while ( (std::cout << prompt) && (!(std::cin >> *var)) )
    {
      std::cin.clear();
      std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // from #include<limits>
    }

  return *var;
}

#endif
