#include "libs/config_parser.h"

using std::cout;
using std::endl;

#include "custom_assert.h"

int main(int argc, char **argv)
{
  Assert(argc == 3, "Usage: prgm <file> <var>");

  Options o(argv[1], Quit, 0);

  cout << o(argv[2]) << endl;

  return 0;
}
