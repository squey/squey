/**
 * @file
 *
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

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
	pvtest::TestEnv env(filename, fileformat);
	env.compute_mapping();

	Inendi::PVView* view = env.compute_plotting().get_parent<Inendi::PVRoot>()->current_view();

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
