#include "String.h"

int String::find(char c, uint pos)
{
  std::string::size_type find_pos;

  find_pos = s.find(c, pos);

  if (find_pos == std::string::npos)
    return -1;

  return find_pos;
}

int   stoi(const std::string &s) {return atoi(s.c_str());}
float stof(const std::string &s) {return atof(s.c_str());}

std::string itos(int i) 
{
  std::stringstream ss;
  ss << i;
  return ss.str();
}

std::string itosNdigits (int number, int N_digits)
{
  int i=0;

  std::string s_str(itos(number)), final;

  while ( i++ < N_digits - (int) s_str.length() )
    final += "0"; // adiciona 0s ao inicio da string
  final += s_str; // coloca-se 
  
  return final;
}
 
