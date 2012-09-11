/**
 * \file PVMappingFilterEnumDefault.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include "PVMappingFilterEnumDefault.h"

#ifdef WIN32
#include <float.h> // for _logb()
#endif
#include <math.h>

Picviz::PVMappingFilterEnumDefault::PVMappingFilterEnumDefault(PVCore::PVArgumentList const& args):
	PVMappingFilter(),
	_case_sensitive(true) // This will be changed by set_args anyway
{
	INIT_FILTER(PVMappingFilterEnumDefault, args);
}

DEFAULT_ARGS_FILTER(Picviz::PVMappingFilterEnumDefault)
{
	PVCore::PVArgumentList args;
	args[PVCore::PVArgumentKey("convert-lowercase", "Convert strings to lower case")].setValue<bool>(false);
	return args;
}

void Picviz::PVMappingFilterEnumDefault::set_args(PVCore::PVArgumentList const& args)
{
	Picviz::PVMappingFilter::set_args(args);
	_case_sensitive = !args["convert-lowercase"].toBool();
}

uint32_t Picviz::PVMappingFilterEnumDefault::_enum_position_factorize(uint32_t v)
{
	// From http://graphics.stanford.edu/~seander/bithacks.html#ReverseByteWith64BitsDiv
	
	// swap odd and even bits
	v = ((v >> 1) & 0x55555555) | ((v & 0x55555555) << 1);
	// swap consecutive pairs
	v = ((v >> 2) & 0x33333333) | ((v & 0x33333333) << 2);
	// swap nibbles ... 
	v = ((v >> 4) & 0x0F0F0F0F) | ((v & 0x0F0F0F0F) << 4);
	// swap bytes
	v = ((v >> 8) & 0x00FF00FF) | ((v & 0x00FF00FF) << 8);
	// swap 2-byte long pairs
	v = ( v >> 16             ) | ( v               << 16);

	return v;
}

Picviz::PVMappingFilter::decimal_storage_type* Picviz::PVMappingFilterEnumDefault::operator()(PVRush::PVNraw::const_trans_nraw_table_line const& values)
{
	if (_case_sensitive) {
		return process<hash_values>(values);
	}
	else {
		return process<hash_nocase_values>(values);
	}
}

IMPL_FILTER(Picviz::PVMappingFilterEnumDefault)
