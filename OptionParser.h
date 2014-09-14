#ifndef OPTION_PARSER_H__
#define OPTION_PARSER_H__

//#include "Buffer.h"

#include <iostream>
#include <vector>
#include <string>

#include <map>

#include "custom_assert.h"

struct NamePair
{
  NamePair(char _Short);                           
  NamePair(const std::string &_Long);              
  NamePair(const std::string &_Long, char _Short); 

  char Short;
  std::string Long;
};

/**
   Usage:
   
   Long names must have size > 1. Short names have length of 1 (char). 
   Strings are converted automatically to short name only if have size == 1.

   Standard program arguments come after all the flags and options.
   The argument to an option comes in the next argv.

   Concatenation of short flags is still not implemented. 

   Example:

     prgm -o -r --window Hamming  -w Hamming other standard program arguments
 */
/// First set options and flags, then parse() the command line and finally get the options and flags.
class OptionParser
{
 public:
 OptionParser() : _ids(1) {} // Start at 1, invalid id values are 0.
  
  void addUsage(const std::string &usage);
  void printUsage();

  void setFlag(char short_flag);
  void setFlag(const std::string &long_flag);
  void setFlag(const std::string &long_flag, char short_flag);

  void setOption(char short_option);
  void setOption(const std::string &long_option);
  void setOption(const std::string &long_option, char short_option);

  bool getFlag(char              short_flag);
  bool getFlag(const std::string &long_flag);

  std::string getOption(char              short_option);
  std::string getOption(const std::string &long_option);

  int parse(int argc, char **argv); // Returns the index of the arg where you should start reading nameless parameters.

  void print();

 private:
  bool valid(char Short);
  bool valid(const std::string &Long);
  
  bool has_flag  (char Short);
  bool has_flag  (const std::string &Long);
  bool has_option(char Short);
  bool has_option(const std::string &Long);

  // Automatically converts to short name if needed.
  int flag_id(const std::string &flag);
  int option_id(const std::string &opt);
  bool set_flag_value(const std::string &flag, bool value);
  bool set_option_value(const std::string &option, const std::string &value);
  
  std::string _usage;

  std::vector<NamePair> _flag_names, _option_names;


  // Each short/long name is associated to an id.
  std::map<char       , int> _short_flag_ids, _short_option_ids;
  std::map<std::string, int>  _long_flag_ids,  _long_option_ids;
  // From the id fetch the value.
  std::map<int, bool>        _flag_values;
  std::map<int, std::string> _option_values;

  int _ids;
};

#endif // OPTION_PARSER_H__
