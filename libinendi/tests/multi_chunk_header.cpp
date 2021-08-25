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

#include "common.h"

#include <pvkernel/core/inendi_assert.h>

#include <pvcop/db/algo.h>

int main()
{
	std::vector<std::string> inputs{TEST_FOLDER "/picviz/inc_1M.csv",
	                                TEST_FOLDER "/picviz/inc_1M.csv"};

	pvtest::TestEnv env(inputs, TEST_FOLDER "/formats/inc.csv.with_header.format", 1);

	const Inendi::PVSource* src = env.root.get_children<Inendi::PVSource>().front();
	PV_VALID(src->get_row_count(), 1999986U);

	std::string sum = pvcop::db::algo::sum(src->get_rushnraw().column(PVCol(1))).at(0);
	PV_VALID(sum, std::string("1000000999944"));

	return 0;
}
