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

#include <pvkernel/core/inendi_assert.h>

#include <pvparallelview/PVQuadTree.h>

#include "common.h"

#include <stdlib.h>

#define VALUE_MAX (1UL << 22)
#define VALUE_MASK (VALUE_MAX - 1)

using quadtree_t = PVParallelView::PVQuadTree<>;
using quadtree_entry_t = PVParallelView::PVQuadTreeEntry;

int main(int argc, char** argv)
{
	size_t num = 1000;
	if (argc >= 2) {
		num = atol(argv[1]);
	} else if (argc == 3) {
		num = atol(argv[1]);
		srand(atoi(argv[2]));
	} else {
		std::cout << "usage: " << basename(argv[0]) << ": num [seed]" << std::endl;
		return 0;
	}

	std::cout << "initialization, it can take a while" << std::endl;
	std::unique_ptr<quadtree_t> qt(new quadtree_t(0, VALUE_MAX, 0, VALUE_MAX, 8));
	for (size_t i = 0; i < num; ++i) {
		qt->insert(quadtree_entry_t(i, random() & VALUE_MASK, random() & VALUE_MASK));
	}
	std::cout << "done" << std::endl;

	return 0;
}
