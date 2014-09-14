#include "OptionParser.h"

//using std:cout;
//using std::endl;
using std::string;

#include <iostream>

using namespace std;


NamePair::NamePair(char _Short)
{ 
  Short = _Short; 
  Long = ""; 
}
NamePair::NamePair(const std::string &_Long)
{
  Guarantee(_Long.size() > 1, "Long name must have size > 1.");
    
  Short = 0;
  Long = _Long;
}
NamePair::NamePair(const std::string &_Long, char _Short)
{ 
  Guarantee(_Long.size() > 1, "Long name must have size > 1.");

  Short = _Short; 
  Long = _Long; 
}


int OptionParser::flag_id(const std::string &flag)
{
  Guarantee(flag.size(), "Empty name!");

  if (flag.size() > 1)
    return _long_flag_ids[flag];
  else
    return _short_flag_ids[flag[0]];
}

int OptionParser::option_id(const std::string &option)
{
  Guarantee(option.size(), "Empty name!");

  if (option.size() > 1)
    return _long_option_ids[option];
  else
    return _short_option_ids[option[0]];
}


bool OptionParser::set_flag_value(const std::string &flag, bool value)
{
  int id = flag_id(flag);
  if (! id)
    return false;

  _flag_values[id] = value;
  return true;
}
bool OptionParser::set_option_value(const std::string &option, const std::string &value)
{
  int id = option_id(option);
  if (! id)
    return false;

  _option_values[id] = value;
  return true;
}

template <class T>
bool has(vector<T> &vec, T value)
{
  for (T v : vec)
    if (value == v)
      return true;

  return false;
}

// Valid for small vectors (int has small range). Returns -1 if it failed.
template <class T>
int index(vector<T> &vec, T value)
{
  for (int i = 0; i < vec.size(); ++i)
    {
      if (value == vec.at(i))
	return i;
    }

  return -1;
}

bool OptionParser::has_flag(char Short)
{
  for (NamePair &name : _flag_names)
    {
      if (name.Short == Short)
	return true;
    }
  return false;
}
bool OptionParser::has_flag(const std::string &Long)
{
  Guarantee(Long.size(), "Empty name!");

  // If needed convert to short name.
  if (Long.size() == 1)
    return has_flag(Long[0]); 

  // Valid long name.
  for (NamePair &name : _flag_names)
    {
      if (name.Long == Long)
	    return true;
    }
  return false;
}

bool OptionParser::has_option(char Short)
{
  for (NamePair &name : _option_names)
    {
      if (name.Short == Short)
	return true;
    }
  return false;
}
bool OptionParser::has_option(const std::string &Long)
{
  Guarantee(Long.size(),"Empty name!");
  
  // If needed convert to short name.
  if (Long.size() == 1)
    return has_option(Long[0]); 

  // Valid long name.
  for (NamePair &name : _option_names)
    {
      if (name.Long == Long)
	return true;
    }
  return false;
}

int OptionParser::parse(int argc, char **argv)
{
  int i = 1; // Skip the program name.
  while (i < argc)
    {
      std::string arg(argv[i]);
      if (arg.substr(0,2) == "--" || arg.substr(0,1) == "-") // Flag / Option
	{
	  if (arg.substr(0,2) == "--") // Long flag/option
	    arg = arg.substr(2);
	  else // Short flag/option
	    arg = arg.substr(1);

	  int pos;
	  
	  if ( has_flag(arg) ) // Flag
	    set_flag_value(arg, true);
	  else if ( has_option(arg) ) // Option
	    {
	      Guarantee(i+1 < argc, "Last option %s wasn't given an argument.", arg.c_str());

	      set_option_value(arg, argv[i+1]);

	      // We already parsed the option argument. Skip argv[i+1].
	      ++i; 
	    }
	  else
	    Guarantee(0, "Used unknown option %s\n", arg.c_str());
	}	  
      else // Standard argument (neither flag or option)
	return i;
	
      ++i;
    }

  return argc;
}

void OptionParser::addUsage(const std::string &usage)
{
  _usage = usage;
}

void OptionParser::printUsage()
{
  cout << _usage;
}

bool OptionParser::valid(char Short)
{
  for (NamePair &name : _flag_names)
    if (name.Short == Short)
      return false;
  
  for (NamePair &name : _option_names)
    if (name.Short == Short)
      return false;

  return true;
}


bool OptionParser::valid(const std::string &Long)
{
  for (NamePair &name : _flag_names)
    if (name.Long == Long)
      return false;
  
  for (NamePair &name : _option_names)
    if (name.Long == Long)
      return false;

  return true;
}

void OptionParser::setFlag(char flag)
{
  Guarantee(valid(flag), "%c was already used.", flag);

  _flag_names.push_back( NamePair(flag) );
  _short_flag_ids[flag] = _ids++;
}
void OptionParser::setFlag(const std::string &flag)
{
  if (flag.size() == 1)
    {
      setFlag(flag[0]);
      return;
    }

  Guarantee(valid(flag), "%s was already used.", flag.c_str());

  _flag_names.push_back( NamePair(flag) );  
  _long_flag_ids[flag] = _ids++;
}
void OptionParser::setFlag(const std::string &long_flag, char short_flag)
{
  Guarantee(valid(long_flag) && valid(short_flag), "%s or %c was already used.", long_flag.c_str(), short_flag);

  _flag_names.push_back( NamePair(long_flag,short_flag) );    
  _long_flag_ids[long_flag] = _short_flag_ids[short_flag] = _ids++;
}


void OptionParser::setOption(char option)
{
  Guarantee(valid(option), "%c was already used.", option);

  _option_names.push_back( NamePair(option) );
  _short_option_ids[option] = _ids++;
}
void OptionParser::setOption(const std::string &option)
{
  if (option.size() == 1)
    {
      setOption(option);
      return;
    }

  Guarantee(valid(option), "%s was already used.", option.c_str());

  _option_names.push_back( NamePair(option) );  
  _long_option_ids[option] = _ids++;
}
void OptionParser::setOption(const std::string &long_option, char short_option)
{
  Guarantee(valid(long_option) && valid(short_option), "%s or %c was already used.", long_option.c_str(), short_option);

  _option_names.push_back( NamePair(long_option,short_option) );    
  _long_option_ids[long_option] = _short_option_ids[short_option] = _ids++;
}

bool OptionParser::getFlag(char short_flag)
{
  int id = _short_flag_ids[short_flag];
  return _flag_values[ id ];    
}
bool OptionParser::getFlag(const std::string &long_flag)
{
  if (long_flag.size() == 1)
    return getFlag(long_flag[0]);

  return _flag_values[ _long_flag_ids[long_flag] ];
}

std::string OptionParser::getOption(char short_option)
{
  return _option_values[ _short_option_ids[short_option] ];
}
std::string OptionParser::getOption(const std::string &long_option)
{
  if (long_option.size() == 1)
    return getOption(long_option[0]);

  return _option_values[ _long_option_ids[long_option] ];
}

bool OptionParser::Flag(char short_flag)
{
  return _flag_values[ _short_flag_ids[short_flag] ];
}
bool OptionParser::Flag(const std::string &long_flag)
{
  if (long_flag.size() == 1)
    return Flag(long_flag[0]);
  return _flag_values[ _long_flag_ids[long_flag] ];
}

bool OptionParser::Option(char short_option)
{
  return _option_values[ _short_option_ids[short_option] ] != "";
}
bool OptionParser::Option(const std::string &long_option)
{
  if (long_option.size() == 1)
    return Option(long_option[0]);
  return _option_values[ _long_option_ids[long_option] ] != "";
}



void OptionParser::print()
{
  puts("");
  if (_flag_names.size())
    {
      printf("Flags: ");
      for (auto &f: _flag_names)
	{
	  int id;
	  if (f.Short)
	    id = _short_flag_ids[f.Short];
	  else
	    id = _long_flag_ids[f.Long];
	  printf("[%s](%c)=%d ", f.Long.c_str(), f.Short, _flag_values[id]);
	  puts("");
	}
      if (_option_names.size())
	{
	  printf("Options: ");
	  for (auto &f: _option_names)
	    {
	      int id;
	      if (f.Short)
		id = _short_option_ids[f.Short];
	      else
		id = _long_option_ids[f.Long];
	      printf("[%s](%c)=<%s> ", f.Long.c_str(), f.Short, _option_values[id].c_str());
	    }
	  puts("");
	}
      puts("");
    }
}
