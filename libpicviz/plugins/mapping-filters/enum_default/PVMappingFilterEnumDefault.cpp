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

uint32_t Picviz::PVMappingFilterEnumDefault::_enum_position_factorize(qlonglong enumber)
{
	uint32_t res = 0;
#ifdef WIN32
	int N = _logb(enumber);
#else
	int N = ilogb(enumber);
#endif

	int i;
	int x = enumber;

	if ( ! enumber) return -1;
	
	for (i = 0; i != N+1; i++) {
		if ((x&1) == 0) {
			res = 2 * res;
		} else {
			res = 1+2*res;
		}
		x = x >> 1;
	}
	
	return res >> (N+1);
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
