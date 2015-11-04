/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <picviz/PVMappingFunction.h>
#include <picviz/PVMappingFactory.h>

#include <stdlib.h>

using Picviz::PVMappingFactory;

int main(void)
{
	// PVMappingFunction mappingfunction("../plugins/functions/libfunction_mapping_enum_default.so");
	PVMappingFactory mf;

#include "test-env.h"

	mf.register_all();

	return 0;
}
