#include "config_parser.h"

using namespace std;

// debug string
#define DS(s) cout << "<" << s << ">" << endl

//! strips the leading and trailing spaces
std::string strip(const std::string &s)
{
  int a=0, b = s.length();

  for (a = 0; s[a]   == ' ' && a<b; ++a) {}
  for (     ; s[b-1] == ' ' && b  ; --b) {}

  return s.substr(a,b-a);
}

Options::Options (const char *filepath, OnFail default_onfail, int print)
{
  std::ifstream cfgfile (filepath);
  std::string line, id, val;
  int eq_pos;

  if (default_onfail == Unspecified)
    _default_onfail = Warn;
  else
    _default_onfail = default_onfail;
  

  if (! cfgfile.is_open())
    throw std::runtime_error(string("Unable to open file: ") + filepath);

  _filepath = filepath;

  if (print)
    cout << "\nConfig file " << filepath << ":\n";
  while(! cfgfile.eof())
    {
      getline(cfgfile, line);
      
      
      line = strip(line);
      if (line[0] == '#' || line == "") 
        continue;  // skip comments

      eq_pos = line.find_first_of("=");

      if (eq_pos < 0) 
        throw std::runtime_error("Config File Parse error: Config file doesn't obey the format.");

      id = strip(line.substr(0,eq_pos));
      val = strip(line.substr(eq_pos+1));
      
      if (print)
	cout << "\t" << id << " = " << val << endl;

      map[id] = val;
    }

  cfgfile.close();
}




const std::string & Options::operator () (const std::string &val, 
					  OnFail onfail) 
{
  if (map[val] == "")
    {
      if (onfail == Unspecified)
	onfail = _default_onfail;

      if (onfail != Ignore)
	{
	  std::cerr << "\nParameter '"
		<< val << "' was not set in file " << _filepath << "." << endl;

	  if (onfail == Quit)
	    exit(1);
	}
    }
  return map[val];
}

int Options::i(const std::string &val, OnFail onfail)
{
  return atoi( ((*this)(val, onfail)).c_str() );
}

float Options::f(const std::string &val, OnFail onfail)
{
  return strtof ( ((*this)(val, onfail)).c_str() , NULL);
}

double Options::d(const std::string &val, OnFail onfail)
{
  return strtod ( ((*this)(val, onfail)).c_str() , NULL );
}


std::map<std::string, std::string> & Options::raw()
{
  return map;
}
