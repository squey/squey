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
#include <pvkernel/rush/PVUtils.h>

#include <fstream>
#include <string>

constexpr const char* filename = TEST_FOLDER "/picviz/honeypot.csv";
constexpr const char* fileformat = TEST_FOLDER "/picviz/honeypot.csv.format";
constexpr const char* ref_out = TEST_FOLDER "/picviz/honeypot.csv.ref";

/**
 * Check format is correctly loaded and mapping/plotting can be computed.
 *
 * @todo : Once mapping is computed after export, we should check it explecitly.
 */
int main()
{
	// Init nraw
	pvtest::TestEnv env(filename, fileformat, 1, pvtest::ProcessUntil::View);

	Inendi::PVView* view = env.root.current_view();

	// Check result
	PVRush::PVNraw const& nraw = view->get_rushnraw_parent();

	std::string out_path = pvtest::get_tmp_filename();
	// Dump the NRAW to file and check value is the same
	nraw.dump_csv(out_path);

	std::cout << out_path << " - " << ref_out << std::endl;
	PV_ASSERT_VALID(PVRush::PVUtils::files_have_same_content(out_path, ref_out));

	std::remove(out_path.c_str());

	return 0;
}
