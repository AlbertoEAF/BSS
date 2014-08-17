#include <iostream>

/// Waits for a ENTER keypress (strong against random input)
void wait()
{
	std::cin.clear();
	std::cin.ignore(9999999999, '\n');
}


