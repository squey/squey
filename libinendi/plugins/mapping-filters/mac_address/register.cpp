/**
 * @file
 *
 * @copyright (C) ESI Group INENDI April 2016
 */

#include <pvbase/export.h>
#include <pvkernel/core/PVClassLibrary.h>

#include "PVMappingFilterMacAddress.h"

LibCPPExport void register_class()
{
	REGISTER_CLASS("mac_address-uni-uni", Inendi::PVMappingFilterMacAddressUU);
	REGISTER_CLASS("mac_address-uni-lin", Inendi::PVMappingFilterMacAddressUL);
	REGISTER_CLASS("mac_address-lin-uni", Inendi::PVMappingFilterMacAddressLU);
	REGISTER_CLASS("mac_address-lin", Inendi::PVMappingFilterMacAddressL);
	REGISTER_CLASS("mac_address-uni", Inendi::PVMappingFilterMacAddressU);
}
