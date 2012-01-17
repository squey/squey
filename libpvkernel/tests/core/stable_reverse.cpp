#include <pvkernel/core/PVAlgorithms.h>
#include <iostream>
#include <utility>
#include <vector>

typedef std::pair<int, int> pair_int_t;
typedef std::vector<pair_int_t> vec_t;

bool less_p(pair_int_t const& v1, pair_int_t const& v2)
{
	return v1.second < v2.second;
}

bool comp(pair_int_t const& v1, pair_int_t const& v2)
{
	return v1.second == v2.second;
}

void dump(vec_t const& v)
{
	vec_t::const_iterator it;
	for (it = v.begin(); it != v.end(); it++) {
		std::cout << it->second << "\t";
	}
	std::cout << std::endl;
	for (it = v.begin(); it != v.end(); it++) {
		std::cout << it->first << "\t";
	}
	std::cout << std::endl;
}

int main()
{
	vec_t v;
	v.resize(5);
	for (int i = 0; i < 5; i++) {
		v[i] = pair_int_t(i, 0);
	}
	dump(v);
	PVCore::stable_sort_reverse(v.begin(), v.end(), comp);
	dump(v);

	std::cout << std::endl << std::endl;

	v.resize(5);
	v[0] = pair_int_t(0, 0);
	v[1] = pair_int_t(0, 0);
	for (int i = 2; i < 5; i++) {
		v[i] = pair_int_t(i, 1);
	}
	dump(v);
	PVCore::stable_sort_reverse(v.begin(), v.end(), comp);
	dump(v);

	std::cout << std::endl << std::endl;

	v.resize(10);
	for (int i = 0; i < 5; i++) {
		v[i] = pair_int_t(i, 0);
	}
	for (int i = 5; i < 10; i++) {
		v[i] = pair_int_t(i, 1);
	}
	dump(v);
	PVCore::stable_sort_reverse(v.begin(), v.end(), comp);
	dump(v);

	std::cout << std::endl << std::endl;

	v.resize(14);
	for (int i = 0; i < 5; i++) {
		v[i] = pair_int_t(i, 0);
	}
	for (int i = 5; i < 14; i++) {
		v[i] = pair_int_t(i, 1);
	}
	dump(v);
	PVCore::stable_sort_reverse(v.begin(), v.end(), comp);
	dump(v);

	std::cout << std::endl << std::endl;

	v.resize(14);
	for (int i = 0; i < 10; i++) {
		v[i] = pair_int_t(i, 0);
	}
	for (int i = 10; i < 14; i++) {
		v[i] = pair_int_t(i, 1);
	}
	dump(v);
	PVCore::stable_sort_reverse(v.begin(), v.end(), comp);
	dump(v);

	std::cout << std::endl << std::endl;
	
	v.resize(20);
	for (int i = 0; i < 10; i++) {
		v[i] = pair_int_t(i, 0);
	}
	for (int i = 10; i < 14; i++) {
		v[i] = pair_int_t(i, 1);
	}
	for (int i = 14; i < 16; i++) {
		v[i] = pair_int_t(i, 5);
	}
	for (int i = 16; i < 20; i++) {
		v[i] = pair_int_t(i, 6);
	}
	dump(v);
	PVCore::stable_sort_reverse(v.begin(), v.end(), comp);
	dump(v);

	std::cout << std::endl << std::endl;

	v.resize(21);
	v[0] = pair_int_t(0, -1);
	for (int i = 1; i < 10; i++) {
		v[i] = pair_int_t(i, 0);
	}
	for (int i = 10; i < 14; i++) {
		v[i] = pair_int_t(i, 1);
	}
	v[14] = pair_int_t(14, -1);
	for (int i = 15; i < 17; i++) {
		v[i] = pair_int_t(i, 5);
	}
	for (int i = 17; i < 21; i++) {
		v[i] = pair_int_t(i, 6);
	}
	std::stable_sort(v.begin(), v.end(), less_p);
	dump(v);
	PVCore::stable_sort_reverse(v.begin(), v.end(), comp);
	dump(v);
	PVCore::stable_sort_reverse(v.begin(), v.end(), comp);
	dump(v);

	return 0;
}
