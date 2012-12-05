/**
 * \file stable_reverse.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/core/PVAlgorithms.h>
#include <iostream>
#include <utility>
#include <vector>

#include <pvkernel/core/picviz_assert.h>

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
	vec_t vref;
	v.resize(5);
	vref.resize(5);
	for (int i = 0; i < 5; i++) {
		v[i] = pair_int_t(i, 0);
		vref[i] = pair_int_t(i, 0);
	}

	std::cout << "test with an already sorted vector" << std::endl;
	bool changed = PVCore::stable_sort_reverse(v.begin(), v.end(), comp);

	PV_VALID(changed, false);
	PV_ASSERT_VALID(v == vref);

	std::cout << "test with an unsorted vector" << std::endl;
	std::cout << "  5 elements" << std::endl;
	v[0] = pair_int_t(0, 0);
	v[1] = pair_int_t(0, 0);
	vref[3] = pair_int_t(0, 0);
	vref[4] = pair_int_t(0, 0);
	for (int i = 2; i < 5; i++) {
		v[i] = pair_int_t(i, 1);
		vref[i - 2] = pair_int_t(i, 1);
	}

	changed = PVCore::stable_sort_reverse(v.begin(), v.end(), comp);
	PV_VALID(changed, true);
	PV_ASSERT_VALID(std::equal (v.begin (), v.end (), vref.begin ()));


	std::cout << "  10 elements" << std::endl;
	v.resize(10);
	vref.resize(10);
	for (int i = 0; i < 5; i++) {
		v[i] = pair_int_t(i, 0);
		vref[i+(10-5)] = pair_int_t(i, 0);
	}
	for (int i = 5; i < 10; i++) {
		v[i] = pair_int_t(i, 1);
		vref[i-5] = pair_int_t(i, 1);
	}
	PVCore::stable_sort_reverse(v.begin(), v.end(), comp);
	PV_ASSERT_VALID(std::equal (v.begin (), v.end (), vref.begin ()));

	std::cout << "  14 elements case 1" << std::endl;
	v.resize(14);
	vref.resize(14);
	for (int i = 0; i < 5; i++) {
		v[i] = pair_int_t(i, 0);
		vref[i+(14-5)] = pair_int_t(i, 0);
	}
	for (int i = 5; i < 14; i++) {
		v[i] = pair_int_t(i, 1);
		vref[i-5] = pair_int_t(i, 1);
	}
	PVCore::stable_sort_reverse(v.begin(), v.end(), comp);
	PV_ASSERT_VALID(std::equal (v.begin (), v.end (), vref.begin ()));

	std::cout << "  14 elements case 2" << std::endl;
	for (int i = 0; i < 10; i++) {
		v[i] = pair_int_t(i, 0);
		vref[i+(14-10)] = pair_int_t(i, 0);
	}
	for (int i = 10; i < 14; i++) {
		v[i] = pair_int_t(i, 1);
		vref[i-10] = pair_int_t(i, 1);
	}
	PVCore::stable_sort_reverse(v.begin(), v.end(), comp);
	PV_ASSERT_VALID(std::equal (v.begin (), v.end (), vref.begin ()));

	std::cout << "  20 elements" << std::endl;
	v.resize(20);
	vref.resize(20);
	for (int i = 0; i < 10; i++) {
		v[i] = pair_int_t(i, 0);
		vref[i+10] = pair_int_t(i, 0);
	}
	for (int i = 10; i < 14; i++) {
		v[i] = pair_int_t(i, 1);
		vref[i-4] = pair_int_t(i, 1);
	}
	for (int i = 14; i < 16; i++) {
		v[i] = pair_int_t(i, 5);
		vref[i-10] = pair_int_t(i, 5);
	}
	for (int i = 16; i < 20; i++) {
		v[i] = pair_int_t(i, 6);
		vref[i-16] = pair_int_t(i, 6);
	}
	PVCore::stable_sort_reverse(v.begin(), v.end(), comp);
	PV_ASSERT_VALID(std::equal (v.begin (), v.end (), vref.begin ()));

	std::cout << "test after a std::stable_sort" << std::endl;
	v.resize(21);
	vref.resize(21);
	v[0] = pair_int_t(0, -1);
	vref[19] = pair_int_t(0, -1);
	for (int i = 1; i < 10; i++) {
		v[i] = pair_int_t(i, 0);
		vref[i+9] = pair_int_t(i, 0);
	}
	for (int i = 10; i < 14; i++) {
		v[i] = pair_int_t(i, 1);
		vref[i-4] = pair_int_t(i, 1);
	}
	v[14] = pair_int_t(14, -1);
	vref[20] = pair_int_t(14, -1);

	for (int i = 15; i < 17; i++) {
		v[i] = pair_int_t(i, 5);
		vref[i-11] = pair_int_t(i, 5);
	}
	for (int i = 17; i < 21; i++) {
		v[i] = pair_int_t(i, 6);
		vref[i-17] = pair_int_t(i, 6);
	}

	std::stable_sort(v.begin(), v.end(), less_p);
	PVCore::stable_sort_reverse(v.begin(), v.end(), comp);
	PV_ASSERT_VALID(std::equal (v.begin (), v.end (), vref.begin ()));

	std::cout << "test after a second call to PVCore::stable_sort_reverse" << std::endl;
	vref[0] = pair_int_t(0, -1);
	vref[1] = pair_int_t(14, -1);
	for (int i = 1; i < 10; i++) {
		vref[i+1] = pair_int_t(i, 0);
	}

	for (int i = 10; i < 14; i++) {
		vref[i+1] = pair_int_t(i, 1);
	}

	for (int i = 15; i < 17; i++) {
		vref[i] = pair_int_t(i, 5);
	}

	for (int i = 17; i < 21; i++) {
		vref[i] = pair_int_t(i, 6);
	}

	PVCore::stable_sort_reverse(v.begin(), v.end(), comp);
	PV_ASSERT_VALID(std::equal (v.begin (), v.end (), vref.begin ()));

	return 0;
}
