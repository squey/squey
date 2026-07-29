//
// MIT License
//
// © Squey, 2026
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

#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/filter/PVChunkFilterByElt.h>
#include <pvkernel/filter/PVElementFilterByFields.h>
#include <pvkernel/core/squey_assert.h>

#include "helpers.h"
#include "common.h"

/*
 * The reverse sub-domain option of the DNS FQDN splitter used to invert a
 * reverse IP into a 64 byte stack buffer without checking how much it wrote,
 * and to read a fixed 63 bytes for the IPv6 form whatever the field held. Both
 * are driven by the imported file, so this covers the inversion path the
 * splitter_dns_fqdn test leaves out (it runs with subd1_rev off).
 */
static const std::vector<std::pair<std::string, std::string>> test_cases = {
    // a well formed reverse IPv4 is inverted back
    {"4.3.2.1.in-addr.arpa", "1.2.3.4"},
    // labels far longer than an octet do not fit once inverted, so the
    // inversion is given up on rather than overflowing the buffer
    {"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.bbbbbbbbbbbbbbbbbbbbbbbbbbbbbb."
     "cccccccccccccccccccccccccccccc.dddddddddddddddddddddddddddddd.in-addr.arpa",
     ""},
    // a reverse IPv6 shorter than the 32 quartets of a full address is only
    // read up to its own length
    {"1.2.ip6.arpa", "2.1"},
};

int main()
{
	const std::string log_file = pvtest::get_tmp_filename();
	{
		std::ofstream ofs{std::filesystem::path(log_file)};
		for (const auto& [input, _] : test_cases) {
			ofs << input << "\n";
		}
	}

	pvtest::TestSplitter ts(log_file);

	PVFilter::PVFieldsSplitter::p_type sp_lib_p =
	    LIB_CLASS(PVFilter::PVFieldsSplitter)::get().get_class_by_name("dns_fqdn");

	// Only the reversed sub-domain, so that each line yields a single field
	PVCore::PVArgumentList args = sp_lib_p->get_args();
	args["tld1"] = false;
	args["tld2"] = false;
	args["tld3"] = false;
	args["subd1"] = true;
	args["subd2"] = false;
	args["subd3"] = false;
	args["subd1_rev"] = true;
	sp_lib_p->set_args(args);

	auto ff = std::make_unique<PVFilter::PVElementFilterByFields>();
	ff->add_filter(sp_lib_p);
	PVFilter::PVChunkFilterByElt chk_flt{std::move(ff)};

	auto res = ts.run_normalization(chk_flt);
	const std::string output_file = std::get<2>(res);

	PV_VALID(std::get<1>(res), std::get<0>(res));

	std::ifstream ifs{std::filesystem::path(output_file)};
	std::string line;
	for (const auto& [input, expected] : test_cases) {
		std::cout << "checking '" << input << "'" << std::endl;
		PV_ASSERT_VALID(bool(std::getline(ifs, line)));
		PV_VALID(line, expected);
	}
	PV_ASSERT_VALID(not bool(std::getline(ifs, line)));

	std::remove(output_file.c_str());
	std::remove(log_file.c_str());
	return 0;
}
