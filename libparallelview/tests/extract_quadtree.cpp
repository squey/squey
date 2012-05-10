
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

#include <pvparallelview/PVHSVColor.h>
#include <pvparallelview/PVBCICode.h>
#include <picviz/PVSelection.h>

// gros hack pour que les quadtree connaissent la structure entry

#pragma pack(push)
#pragma pack(4)

struct entry {
	uint32_t y1, y2;
	uint32_t idx;

	bool operator==(const entry &e)
	{
		return ((y1 == e.y1) || (y2 == e.y2) || (idx == e.idx));
	}
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

#define MAX_VALUE ((1<<22) - 1)

void print_mem (const char *text, size_t s)
{
	double v = s / (1024. * 1024.);
	std::cout << text  << ": memory usage is: " << v << " Mib" << std::endl;
}

enum {
	TEST_FIRST_Y1_FULL = 0,
	TEST_FIRST_Y1Y2_FULL,
	TEST_FIRST_SEL_FULL,
	TEST_FIRST_SEL_HALF,
	TEST_FIRST_SEL_QUARTER,
	TEST_FIRST_SEL_NONE,

	TEST_FIRST_BCI_Y1_FULL,
	TEST_FIRST_BCI_Y1Y2_FULL,
	TEST_FIRST_BCI_SEL_FULL,
	TEST_FIRST_BCI_SEL_HALF,
	TEST_FIRST_BCI_SEL_QUARTER,
	TEST_FIRST_BCI_SEL_NONE,

	TEST_SUB_Y1_FULL,
	TEST_SUB_Y1_HALF,
	TEST_SUB_Y1Y2_FULL,
	TEST_SUB_Y1Y2_QUARTER,
	TEST_SUB_Y1Y2_FOUR_QUARTER,

	TEST_SUB_SEL_FULL,
	TEST_SUB_SEL_HALF,
	TEST_SUB_SEL_QUARTER,
	TEST_SUB_SEL_NONE,

	TEST_LAST
};

const char *test_text[] = {
	"PVQuadTree::extract_first_y1 with full area",
	"PVQuadTree::extract_first_y1y2 with full area",
	"PVQuadTree::extract_first_selection with full selection",
	"PVQuadTree::extract_first_selection with entry-count / 2 selected entries",
	"PVQuadTree::extract_first_selection with entry-count / 4 selected entries",
	"PVQuadTree::extract_first_selection with no selected entry",

	"PVQuadTree::extract_first_bci_y1 with full area",
	"PVQuadTree::extract_first_bci_y1y2 with full area",
	"PVQuadTree::extract_first_bci_selection with full selection",
	"PVQuadTree::extract_first_bci_selection with entry-count / 2 selected entries",
	"PVQuadTree::extract_first_bci_selection with entry-count / 4 selected entries",
	"PVQuadTree::extract_first_bci_selection with no selected entry",

	"PVQuadTree::extract_subtree_y1 with full area",
	"PVQuadTree::extract_subtree_y1 with half area",
	"PVQuadTree::extract_subtree_y1y2 with full area",
	"PVQuadTree::extract_subtree_y1y2 with a quarter of area",
	"PVQuadTree::extract_subtree_y1y2 with a quarter of area for each quarter",

	"PVQuadTree::extract_subtree_with_selection with full selected entries",
	"PVQuadTree::extract_subtree_with_selection with entry-count / 2 selected entries",
	"PVQuadTree::extract_subtree_with_selection with entry-count / 4 selected entries",
	"PVQuadTree::extract_subtree_with_selection with no selected entry"
};

void usage()
{
	std::cout << "usage: test-quadtree entry-count what" << std::endl;
	std::cout << std::endl;
	std::cout << "what can be:" << std::endl;
	for(unsigned i = 0; i < TEST_LAST; ++i) {
		std::cout << "  " << i << ": " << test_text[i] << std::endl;
	}
	std::cout << std::endl;
}

// it's 8 because QuadTreeTmpl's size can not set
#define DEPTH 8

// lots of global variables O:-)
unsigned count;
int what;
entry *entries;
std::vector<entry> res1;
Picviz::PVSelection *selection;
PVQuadTree<Vector1<entry>, entry> *sqt1;
PVQuadTree<Vector1<entry>, entry> *subtree;
std::vector<PVParallelView::PVBCICode> codes;

// forward declarations
void do_extract_first_tests();
void do_extract_first_bci_tests();
void do_subtree_tests();
void do_selection_tests();

int main(int argc, char **argv)
{
	if (argc != 3) {
		usage();
		return 1;
	}

	count = (unsigned)atoi(argv[1]);
	what = atoi(argv[2]);

	if(what >= TEST_LAST) {
		usage();
		return 2;
	}

	if(what < 0) {
		std::cout << TEST_LAST - 1 << std::endl;
		return 0;
	}

	entries = new entry [count];
	for(unsigned i = 0; i < count; ++i) {
		entries[i].y1 = random() & MAX_VALUE;
		entries[i].y2 = random() & MAX_VALUE;
		entries[i].idx = i;
	}

	selection = new Picviz::PVSelection();

	sqt1 = new PVQuadTree<Vector1<entry>, entry>(0, MAX_VALUE, 0, MAX_VALUE, DEPTH);
	std::cout << "Filling quadtree, it can take a while..." << std::endl;
	for(unsigned i = 0; i < count; ++i) {
		sqt1->insert(entries[i]);
	}

	do_extract_first_tests();
	do_extract_first_bci_tests();
	do_subtree_tests();
	do_selection_tests();

	if(sqt1) {
		delete sqt1;
	}

	if(selection) {
		delete selection;
	}

	return 0;
}

void do_extract_first_tests()
{
	/* worst case of y1 extraction
	 */
	if(what == TEST_FIRST_Y1_FULL) {
		std::cout << "# " << test_text[what] << std::endl;
		res1.reserve(0);
		BENCH_START(time);
		sqt1->extract_first_from_y1(0, MAX_VALUE, res1);
		BENCH_END(time, "time", 1, 1, 1, 1);
		std::cout << "search result size : " << res1.size() << std::endl;
	}

	/* worst case of y1y2 extraction
	 */
	if(what == TEST_FIRST_Y1Y2_FULL) {
		std::cout << "# " << test_text[what] << std::endl;
		res1.reserve(0);
		BENCH_START(time);
		sqt1->extract_first_from_y1y2(0, MAX_VALUE, 0, MAX_VALUE, res1);
		BENCH_END(time, "time", 1, 1, 1, 1);
		std::cout << "search result size : " << res1.size() << std::endl;
	}

	if(what == TEST_FIRST_SEL_FULL) {
		std::cout << "# " << test_text[what] << std::endl;
		selection->select_all();
		BENCH_START(time);
		sqt1->extract_first_from_selection(*selection, res1);
		BENCH_END(time, "time", 1, 1, 1, 1);
		std::cout << "search result size : " << res1.size() << std::endl;
	}

	if(what == TEST_FIRST_SEL_HALF) {
		std::cout << "# " << test_text[what] << std::endl;
		selection->select_even();
		BENCH_START(time);
		sqt1->extract_first_from_selection(*selection, res1);
		BENCH_END(time, "time", 1, 1, 1, 1);
		std::cout << "search result size : " << res1.size() << std::endl;
	}

	if(what == TEST_FIRST_SEL_QUARTER) {
		std::cout << "# " << test_text[what] << std::endl;
		memset(selection->get_buffer(), 0x88, PICVIZ_SELECTION_NUMBER_OF_CHUNKS);
		BENCH_START(time);
		sqt1->extract_first_from_selection(*selection, res1);
		BENCH_END(time, "time", 1, 1, 1, 1);
		std::cout << "search result size : " << res1.size() << std::endl;
	}

	if(what == TEST_FIRST_SEL_NONE) {
		std::cout << "# " << test_text[what] << std::endl;
		selection->select_none();
		BENCH_START(time);
		sqt1->extract_first_from_selection(*selection, res1);
		BENCH_END(time, "time", 1, 1, 1, 1);
		std::cout << "search result size : " << res1.size() << std::endl;
	}
}

void do_extract_first_bci_tests()
{
	/* worst case of 1D first BCICode extraction
	 */
	if(what == TEST_FIRST_BCI_Y1_FULL) {
		std::cout << "# " << test_text[what] << std::endl;
		BENCH_START(time);
		sqt1->extract_first_bci_from_y1(0, MAX_VALUE, codes);
		BENCH_END(time, "time", 1, 1, 1, 1);
		std::cout << "search result size : " << codes.size() << std::endl;
	}

	/* worst case of 2D first BCICode extraction
	 */
	if(what == TEST_FIRST_BCI_Y1Y2_FULL) {
		std::cout << "# " << test_text[what] << std::endl;
		BENCH_START(time);
		sqt1->extract_first_bci_from_y1y2(0, MAX_VALUE, 0, MAX_VALUE, codes);
		BENCH_END(time, "time", 1, 1, 1, 1);
		std::cout << "extraction result size : " << codes.size() << std::endl;
	}

	if(what == TEST_FIRST_BCI_SEL_FULL) {
		std::cout << "# " << test_text[what] << std::endl;
		selection->select_all();
		BENCH_START(time);
		sqt1->extract_first_bci_from_selection(*selection, codes);
		BENCH_END(time, "time", 1, 1, 1, 1);
		std::cout << "search result size : " << codes.size() << std::endl;
	}

	if(what == TEST_FIRST_BCI_SEL_HALF) {
		std::cout << "# " << test_text[what] << std::endl;
		selection->select_even();
		BENCH_START(time);
		sqt1->extract_first_bci_from_selection(*selection, codes);
		BENCH_END(time, "time", 1, 1, 1, 1);
		std::cout << "search result size : " << codes.size() << std::endl;
	}

	if(what == TEST_FIRST_BCI_SEL_QUARTER) {
		std::cout << "# " << test_text[what] << std::endl;
		memset(selection->get_buffer(), 0x88, PICVIZ_SELECTION_NUMBER_OF_CHUNKS);
		BENCH_START(time);
		sqt1->extract_first_bci_from_selection(*selection, codes);
		BENCH_END(time, "time", 1, 1, 1, 1);
		std::cout << "search result size : " << codes.size() << std::endl;
	}

	if(what == TEST_FIRST_BCI_SEL_NONE) {
		std::cout << "# " << test_text[what] << std::endl;
		selection->select_none();
		BENCH_START(time);
		sqt1->extract_first_bci_from_selection(*selection, codes);
		BENCH_END(time, "time", 1, 1, 1, 1);
		std::cout << "search result size : " << res1.size() << std::endl;
	}
}

void do_subtree_tests()
{

	/* comparison with a full extraction
	 */
	if(what == TEST_SUB_Y1_FULL) {
		std::cout << "# " << test_text[what] << std::endl;
		BENCH_START(time);
		subtree = sqt1->extract_subtree_y1(0, MAX_VALUE);
		BENCH_END(time, "time", 1, 1, 1, 1);
		print_mem("QuadTree", sqt1->memory());
		print_mem("SubQuadTree", subtree->memory());
		if(*sqt1 == *subtree) {
			std::cout << "subtree is equal" << std::endl;
		} else {
			std::cout << "subtree is different" << std::endl;
		}
		delete subtree;
	}

	if(what == TEST_SUB_Y1_HALF) {
		std::cout << "# " << test_text[what] << std::endl;
		BENCH_START(time);
		subtree = sqt1->extract_subtree_y2(0, MAX_VALUE >> 1);
		BENCH_END(time, "time", 1, 1, 1, 1);
		print_mem("QuadTree", sqt1->memory());
		print_mem("SubQuadTree", subtree->memory());
		std::cout << "QuadTree's elements count: " << sqt1->elements() << std::endl;
		std::cout << "SubQuadTree's elements count: " << subtree->elements() << std::endl;
		delete subtree;
	}

	if(what == TEST_SUB_Y1Y2_FULL) {
		std::cout << "# " << test_text[what] << std::endl;
		BENCH_START(time);
		subtree = sqt1->extract_subtree_y1y2(0, MAX_VALUE,
		                                     0, MAX_VALUE);
		BENCH_END(time, "time", 1, 1, 1, 1);
		print_mem("QuadTree", sqt1->memory());
		print_mem("SubQuadTree", subtree->memory());
		std::cout << "QuadTree's elements count: " << sqt1->elements() << std::endl;
		std::cout << "SubQuadTree's elements count: " << subtree->elements() << std::endl;
		delete subtree;
	}

	if(what == TEST_SUB_Y1Y2_QUARTER) {
		std::cout << "# " << test_text[what] << std::endl;
		BENCH_START(time);
		subtree = sqt1->extract_subtree_y1y2(0, (MAX_VALUE >> 1),
		                                     0, (MAX_VALUE >> 1));
		BENCH_END(time, "time", 1, 1, 1, 1);
		print_mem("QuadTree", sqt1->memory());
		print_mem("SubQuadTree", subtree->memory());
		std::cout << "QuadTree's elements count: " << sqt1->elements() << std::endl;
		std::cout << "SubQuadTree's elements count: " << subtree->elements() << std::endl;
		delete subtree;
	}

	if(what == TEST_SUB_Y1Y2_FOUR_QUARTER) {
		std::cout << "# " << test_text[what] << std::endl;
		{
			// SW quarter
			BENCH_START(time);
			subtree = sqt1->extract_subtree_y1y2(0, MAX_VALUE >> 1,
			                                     0, MAX_VALUE >> 1);
			BENCH_END(time, "time", 1, 1, 1, 1);
			print_mem("QuadTree", sqt1->memory());
			print_mem("SubQuadTree", subtree->memory());
			std::cout << "QuadTree's elements count: " << sqt1->elements() << std::endl;
			std::cout << "SubQuadTree's elements count: " << subtree->elements() << std::endl;
			delete subtree;
		}
		{
			// SE quarter
			BENCH_START(time);
			subtree = sqt1->extract_subtree_y1y2(0             , MAX_VALUE >> 1,
			                                     MAX_VALUE >> 1, MAX_VALUE);
			BENCH_END(time, "time", 1, 1, 1, 1);
			print_mem("QuadTree", sqt1->memory());
			print_mem("SubQuadTree", subtree->memory());
			std::cout << "QuadTree's elements count: " << sqt1->elements() << std::endl;
			std::cout << "SubQuadTree's elements count: " << subtree->elements() << std::endl;
			delete subtree;
		}
		{
			// NE quarter
			BENCH_START(time);
			subtree = sqt1->extract_subtree_y1y2(MAX_VALUE >> 1, MAX_VALUE,
			                                     MAX_VALUE >> 1, MAX_VALUE);
			BENCH_END(time, "time", 1, 1, 1, 1);
			print_mem("QuadTree", sqt1->memory());
			print_mem("SubQuadTree", subtree->memory());
			std::cout << "QuadTree's elements count: " << sqt1->elements() << std::endl;
			std::cout << "SubQuadTree's elements count: " << subtree->elements() << std::endl;
			delete subtree;
		}
		{
			// NW quarter
			BENCH_START(time);
			subtree = sqt1->extract_subtree_y1y2(MAX_VALUE >> 1, MAX_VALUE,
			                                     0             , MAX_VALUE >> 1);
			BENCH_END(time, "time", 1, 1, 1, 1);
			print_mem("QuadTree", sqt1->memory());
			print_mem("SubQuadTree", subtree->memory());
			std::cout << "QuadTree's elements count: " << sqt1->elements() << std::endl;
			std::cout << "SubQuadTree's elements count: " << subtree->elements() << std::endl;
			delete subtree;
		}
	}
}

void do_selection_tests()
{

	if(what == TEST_SUB_SEL_FULL) {
		std::cout << "# " << test_text[what] << std::endl;
		selection->select_all();
		BENCH_START(time);
		subtree = sqt1->extract_subtree_from_selection(*selection);
		BENCH_END(time, "time", 1, 1, 1, 1);
		print_mem("QuadTree", sqt1->memory());
		print_mem("SubQuadTree", subtree->memory());
		if(*sqt1 == *subtree) {
			std::cout << "subtree is equal" << std::endl;
		} else {
			std::cout << "subtree is different" << std::endl;
		}
		delete subtree;
	}

	if(what == TEST_SUB_SEL_HALF) {
		std::cout << "# " << test_text[what] << std::endl;
		selection->select_even();
		BENCH_START(time);
		subtree = sqt1->extract_subtree_from_selection(*selection);
		BENCH_END(time, "time", 1, 1, 1, 1);
		print_mem("QuadTree", sqt1->memory());
		print_mem("SubQuadTree", subtree->memory());
		std::cout << "QuadTree's elements count: " << sqt1->elements() << std::endl;
		std::cout << "SubQuadTree's elements count: " << subtree->elements() << std::endl;
		delete subtree;
	}

	if(what == TEST_SUB_SEL_QUARTER) {
		std::cout << "# " << test_text[what] << std::endl;
		memset(selection->get_buffer(), 0x88, PICVIZ_SELECTION_NUMBER_OF_BYTES);
		BENCH_START(time);
		subtree = sqt1->extract_subtree_from_selection(*selection);
		BENCH_END(time, "time", 1, 1, 1, 1);
		print_mem("QuadTree", sqt1->memory());
		print_mem("SubQuadTree", subtree->memory());
		std::cout << "QuadTree's elements count: " << sqt1->elements() << std::endl;
		std::cout << "SubQuadTree's elements count: " << subtree->elements() << std::endl;
		delete subtree;
	}

	if(what == TEST_SUB_SEL_NONE) {
		std::cout << "# " << test_text[what] << std::endl;
		selection->select_none();
		BENCH_START(time);
		subtree = sqt1->extract_subtree_from_selection(*selection);
		BENCH_END(time, "time", 1, 1, 1, 1);
		print_mem("QuadTree", sqt1->memory());
		print_mem("SubQuadTree", subtree->memory());
		std::cout << "QuadTree's elements count: " << sqt1->elements() << std::endl;
		std::cout << "SubQuadTree's elements count: " << subtree->elements() << std::endl;
		delete subtree;
	}
}
