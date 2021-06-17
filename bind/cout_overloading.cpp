#include "cout_overloading.h"
#include <iostream>
using namespace std;

std::ostream& operator <<(std::ostream& o, const std::exception& e) {
	o << e.what() << endl;
	return o;
}