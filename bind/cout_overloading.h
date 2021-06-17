#pragma once
#include <vector>
#include <iostream>
#include <exception>

template<class T>
std::ostream& operator <<(std::ostream& o, const std::vector<T>& v) {
	o << "[";
	for (int i = 0; i < v.size(); i++) {
		if (i < v.size() - 1) {
			o << v[i] << ",";
		}
		else {
			o << v[i];
		}
	}
	o << "]";
	return o;
}

std::ostream& operator <<(std::ostream& o, const std::exception& e);