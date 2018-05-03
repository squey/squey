/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2018
 */

#include "../rush/common.h"

#include <pvbase/general.h>
#include <pvkernel/core/PVCrashReportSender.h>
#include <pvkernel/core/PVLicenseActivator.h>
#include <pvkernel/core/inendi_assert.h>

#include <algorithm>

int main()
{
	std::string minidump_path = pvtest::get_tmp_filename();
	std::ofstream out(minidump_path, std::ofstream::trunc);
	out << "this is the crash report data" << std::endl;
	std::string version = std::string(INENDI_CURRENT_VERSION_STR) + "_dry-run";

	bool ret1 =
	    PVCore::PVCrashReportSender::send(minidump_path, version, "1111-*111 1111 1111 1111"); // OK
	bool ret2 =
	    PVCore::PVCrashReportSender::send(minidump_path, version, "!!!!-*!!! !!!! !!!! !!!!"); // KO

	std::remove(minidump_path.c_str());

	PV_ASSERT_VALID(ret1);
	PV_ASSERT_VALID(not ret2);

	return 0;
}
