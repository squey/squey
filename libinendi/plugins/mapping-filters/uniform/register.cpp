/**
 * @file
 *
 * @copyright (C) ESI Group INENDI April 2016
 */

#include <pvbase/export.h>
#include <pvkernel/core/PVClassLibrary.h>

#include "PVMappingFilterUniform.h"

LibCPPExport void register_class()
{
	REGISTER_CLASS("uniform", Inendi::PVMappingFilterUniform);
}
