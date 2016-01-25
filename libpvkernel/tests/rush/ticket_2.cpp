/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/filter/PVPluginsLoad.h>
#include <pvkernel/rush/PVPluginsLoad.h>
#include <pvkernel/rush/PVExtractor.h>
#include <pvkernel/rush/PVControllerJob.h>
#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/rush/PVTests.h>
#include <pvkernel/rush/PVFileDescription.h>
#include <cstdlib>
#include <iostream>
#include "helpers.h"

#include "common.h"

int main()
{
	pvtest::TestEnv env(TEST_FOLDER "/pvkernel/rush/tickets/2/apache.access", TEST_FOLDER "/pvkernel/rush/tickets/2/apache.access.format");

	// Ask for 1 million lines
	env.load_data(1000000);

	return 0;
}
