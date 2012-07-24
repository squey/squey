/**
 * \file create_quadtree.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <iostream>
#include <vector>
#include <algorithm>

#include <stdint.h>
#include <limits.h>
#include <stdlib.h>

#include <boost/random.hpp>
#include <boost/random/uniform_int_distribution.hpp>

#include <boost/math/distributions/normal.hpp>

#include <pvkernel/core/picviz_bench.h>

// gros hack pour que les quadtree connaissent la structure entry

#pragma pack(push)
#pragma pack(4)

struct entry {
	uint32_t y1, y2;
	uint32_t idx;
};

bool are_diff(const entry &e1, const entry &e2)
{
	return ((e1.y1 != e2.y1) || (e1.y2 != e2.y2) || (e1.idx != e2.idx));
}

#pragma pack(pop)

enum {
	SW = 0,
	SE,
	NW,
	NE
};

#include "quadtree.h"
#include "quadtree-tmpl.h"
#include "quadtree-flat.h"

void printb (uint32_t v)
{
	for(int i = 31; i >= 0; --i) {
		if((i & 7) == 7)
			std::cout << " ";
		if(v & (1 << i))
			std::cout << "1";
		else
			std::cout << "0";
	}
}

void print_mem (const char *text, size_t s)
{
	double v = s / (1024. * 1024.);
	std::cout << text  << ": memory usage is: " << v << " Mib" << std::endl;
}

#define MAX_VALUE ((1<<22) - 1)

void usage()
{
	std::cout << "usage: test-quadtree entry-count test-num1 test-num2..." << std::endl;
	std::cout << std::endl;
	std::cout << "test-num can be:" << std::endl;
	std::cout << "  0: PVQuadTree with Vector1" << std::endl;
	std::cout << "  1: PVQuadTreeTmpl with Vector1" << std::endl;
	std::cout << "  2: PVQuadTreeFlat with Vector1" << std::endl;
}

#define TESTS_CHECK(vec, value) (std::find(vec.begin(), vec.end(), value) != vec.end())

// it's 8 because QuadTreeTmpl's size can not set
#define DEPTH 8

int main(int argc, char **argv)
{
	if (argc < 3) {
		usage();
		return 1;
	}

	int count = atoi(argv[1]);

	std::vector<int> tests;

	for (int i = 2; i < argc; ++i) {
		tests.push_back(atoi(argv[i]));
	}

	boost::mt19937 rnd(0);
	boost::random::uniform_int_distribution<unsigned> uni(0, UINT_MAX);

	entry *entries = new entry [count];
	for(int i = 0; i < count; ++i) {
		entries[i].y1 = random() & MAX_VALUE;
		entries[i].y2 = random() & MAX_VALUE;
		entries[i].idx = i;
	}

	entry *ranges = new entry [count];
	for(int i = 0; i < count; ++i) {
		uint32_t y1 = random() & MAX_VALUE;
		uint32_t y2 = random() & MAX_VALUE;
		if(y1 < y2) {
			ranges[i].y1 = y1;
			ranges[i].y2 = y2;
		} else {
			ranges[i].y1 = y2;
			ranges[i].y2 = y1;
		}
	}

	PVQuadTree<Vector1<entry>, entry> *sqt1 = 0;
	if (TESTS_CHECK(tests, 0)) {
		sqt1 = new PVQuadTree<Vector1<entry>, entry>(0, MAX_VALUE, 0, MAX_VALUE, DEPTH);
		MEM_START(usage);
		BENCH_START(time);
		for(int i = 0; i < count; ++i) {
			sqt1->insert(entries[i]);
		}
		BENCH_END(time, "PVQuadTree", count, sizeof(entry), 1, 1);
		MEM_END(usage, "PVQuadTree");
		print_mem("PVQuadTree", sqt1->memory());

		std::vector<entry> res1;
		BENCH_START(time_search_rec);
		sqt1->extract_first_from_y1(0, MAX_VALUE, res1);
		BENCH_END(time_search_rec, "PVQuadTree::extract_first_from_y1", 1, 1, 1, 1);
		std::cout << "search result size : " << res1.size() << std::endl;

		std::vector<entry> res2;
		BENCH_START(time_search_rec2);
		sqt1->extract_first_from_y1y2(0, MAX_VALUE, 0, MAX_VALUE, res2);
		BENCH_END(time_search_rec2, "PVQuadTree::extract_first_from_y1y2", 1, 1, 1, 1);
		std::cout << "search result size : " << res2.size() << std::endl;
	}

	PVQuadTreeTmpl<Vector1<entry>, entry, 8> *tqt1 = 0;
	if (TESTS_CHECK(tests, 1)) {
		tqt1 = new PVQuadTreeTmpl<Vector1<entry>, entry, 8>(0, MAX_VALUE, 0, MAX_VALUE, DEPTH);
		MEM_START(usage);
		BENCH_START(time);
		for(int i = 0; i < count; ++i) {
			tqt1->insert(entries[i]);
		}
		BENCH_END(time, "PVQuadTreeTmpl", count, sizeof(entry), 1, 1);
		MEM_END(usage, "PVQuadTreeTmpl");
		print_mem("PVQuadTreeTmpl", tqt1->memory());
	}

	PVQuadTreeFlat<Vector1<entry>, entry> *fqt1 = 0;
	if (TESTS_CHECK(tests, 2)) {
		fqt1 = new PVQuadTreeFlat<Vector1<entry>, entry>(0, MAX_VALUE, 0, MAX_VALUE, DEPTH);
		MEM_START(usage);
		BENCH_START(time);
		for(int i = 0; i < count; ++i) {
			fqt1->insert(entries[i]);
		}
		BENCH_END(time, "PVQuadTreeFlat", count, sizeof(entry), 1, 1);
		MEM_END(usage, "PVQuadTreeFlat");
		print_mem("PVQuadTreeFlat", fqt1->memory());
	}

	if (sqt1 && tqt1) {
		std::cout << "comparing PVQuadTreeTmpl<Vector1> with PVQuadTree<Vector1>" << std::endl;
		if (tqt1->compare(*sqt1)) {
			std::cout << "    equal" << std::endl;
		} else {
			std::cout << "    not equal" << std::endl;
		}
	}

	if (sqt1 && fqt1) {
		std::cout << "comparing PVQuadTreeTmpl<Vector1> with PVQuadTree<Vector1>" << std::endl;
		if (fqt1->compare(*sqt1)) {
			std::cout << "    equal" << std::endl;
		} else {
			std::cout << "    not equal" << std::endl;
		}
	}

	if(sqt1) {
		delete sqt1;
	}

	if(tqt1) {
		delete tqt1;
	}

	if(fqt1) {
		delete fqt1;
	}

	return 0;
}
