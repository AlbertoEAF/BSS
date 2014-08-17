/*
  In C++11 there are new utils to strings that allow conversion to other types, thus making this 
  +- useless

  I am changing the naming convention to allow smoother transition to the STL in the future

  Though there are some utilities here that are useful, like an easier to use find 
 */

#ifndef String_H__
#define String_H__

#include <fstream>
#include <stdlib.h> // atoi, atof, system
#include <sstream> // for itos, ftos, etc. functions

class String: public std::string {
    
   // friend void operator>> (std::istream &, String &);
 public:
  String(std::string string="") : s(string) {}
     
  int   stoi  () { return atoi(s.c_str()); }
  float stof  () { return atof(s.c_str()); }

  int find (char c, uint pos); // Like std::string:find but returns -1 if no match is found
  uint length() { return s.length(); }

  String substr(uint start, uint length) {return s.substr(start,length);}

  std::string s;
};


int find(char c, uint pos);

int   stoi(const std::string &s);
float stof(const std::string &s);

std::string itos(int i);
std::string itosNdigits (int number, int N_digits);



#endif
