/**
 * @file
 *
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include "common.h"

#include <pvkernel/core/inendi_assert.h>

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
    pvtest::TestEnv env(filename, fileformat);

    Inendi::PVView* view = env.view;

    // Check result
    PVRush::PVNraw const& nraw = view->get_rushnraw_parent();

	std::string out_path = pvtest::get_tmp_filename();
	// Dump the NRAW to file and check value is the same
    nraw.dump_csv(out_path);

	std::ifstream ifs_res(out_path);
	std::string content_res{std::istreambuf_iterator<char>(ifs_res), std::istreambuf_iterator<char>()};

	std::ifstream ifs_ref(ref_out);
	std::string content_ref{std::istreambuf_iterator<char>(ifs_ref), std::istreambuf_iterator<char>()};

	PV_VALID(content_ref, content_res);

	std::remove(out_path.c_str());

    return 0;
}


