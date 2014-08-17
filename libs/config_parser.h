#ifndef CONFIG_PARSER_H__
#define CONFIG_PARSER_H__

#include <string>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <map>
#include <stdlib.h> // atoi, atof, exit, strtod
#include "safe.h"

std::string strip(const std::string &s);

class Options {
 public:
  
  Options(const char *filepath, OnFail default_onfail, int print);


  const std::string & operator () (const std::string &val, 
				   OnFail onfail = Unspecified);

  int    i(const std::string &val, OnFail onfail = Unspecified); 
  float  f(const std::string &val, OnFail onfail = Unspecified); 
  double d(const std::string &val, OnFail onfail = Unspecified); 

  std::map<std::string, std::string> &raw();

 private:

  std::map<std::string, std::string> map;

  OnFail _default_onfail;
  std::string _filepath; // only used for print + debug
};



//Options parse_config(const char *filepath);

#endif
