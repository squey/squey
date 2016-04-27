/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */
#ifndef LIBPVPARALLELVIEW_TESTS_TEST_ENV_H
#define LIBPVPARALLELVIEW_TESTS_TEST_ENV_H

#include <cstdlib>
#include <pvbase/general.h>

static void init_env()
{
	setenv("PVKERNEL_PLUGIN_PATH", INENDI_BUILD_DIRECTORY "/libpvkernel/plugins", 0);
	setenv("INENDI_PLUGIN_PATH", INENDI_BUILD_DIRECTORY "/libinendi/plugins", 0);
}

#endif
