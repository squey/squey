//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <iostream>
#include <vector>

#include <pvkernel/core/inendi_bench.h>

#include <pvparallelview/PVQuadTree.h>

unsigned count;
unsigned depth;

typedef PVParallelView::PVQuadTree<10000, 1000, 10000> pvquadtree;

pvquadtree* qt = nullptr;
PVParallelView::PVQuadTreeEntry* entries = nullptr;

#define MAX_VALUE ((1 << 22) - 1)

void usage()
{
	std::cout << "usage: test-quadtree depth count" << std::endl;
}

int main(int argc, char** argv)
{
	if (argc != 3) {
		usage();
		return 1;
	}

	depth = (unsigned)atoi(argv[1]);
	count = (unsigned)atoi(argv[2]);

	entries = new PVParallelView::PVQuadTreeEntry[count];
	for (unsigned i = 0; i < count; ++i) {
		entries[i].y1 = random() & MAX_VALUE;
		entries[i].y2 = random() & MAX_VALUE;
		entries[i].idx = i;
	}

	qt = new pvquadtree(0, MAX_VALUE, 0, MAX_VALUE, depth);

	std::cout << "Filling quadtree, it can take a while..." << std::endl;
	BENCH_START(fill);
	for (unsigned i = 0; i < count; ++i) {
		qt->insert(entries[i]);
	}
	BENCH_END(fill, "fill", 1, 1, 1, 1);

	std::cout << "sizeof(node): " << sizeof(*qt) << std::endl;
	std::cout << "memory used : " << qt->memory() << std::endl;

	std::unique_ptr<PVParallelView::pv_quadtree_buffer_entry_t> buffer_uptr(new PVParallelView::pv_quadtree_buffer_entry_t[QUADTREE_BUFFER_SIZE]);
	PVParallelView::pv_quadtree_buffer_entry_t* buffer = buffer_uptr.get();
	std::unique_ptr<pvquadtree::pv_tlr_buffer_t> tlr(new pvquadtree::pv_tlr_buffer_t);

	for (unsigned i = 1; i < 9; ++i) {
		size_t num = 0;
		std::cout << "extract BCI codes from y1 for zoom " << i << std::endl;
		BENCH_START(extract);
		qt->get_first_from_y1(
		    0, MAX_VALUE >> i, i, 1, buffer,
		    [&](const PVParallelView::PVQuadTreeEntry& e, pvquadtree::pv_tlr_buffer_t& buffer) {
			    (void)e;
			    (void)buffer;
			},
		    *tlr);
		BENCH_END(extract, "extract", 1, 1, 1, 1);
		std::cout << "elements found: " << num << std::endl;
	}

	std::cout << std::endl;

	if (qt) {
		delete qt;
	}

	return 0;
}
