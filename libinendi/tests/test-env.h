/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <cstdlib>
#include <inendi/common.h>
#include <pvbase/general.h>
#include <pvkernel/core/inendi_intrin.h>

void init_env()
{
	PVCore::PVIntrinsics::init_cpuid();
	setenv("PVKERNEL_PLUGIN_PATH",INENDI_BUILD_DIRECTORY "/libpvkernel/plugins", 0);
	setenv("INENDI_PLUGIN_PATH",INENDI_BUILD_DIRECTORY "/libinendi/plugins", 0);
	Inendi::common::load_filters();
}
