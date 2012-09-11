/**
 * \file PVMappingFilterStringDefault.cpp
 *
 * Copyright (C) Picviz Labs 2011-2012
 */

#include "PVMappingFilterStringDefault.h"
#include <pvkernel/core/PVTBBMaxArray.h>
#include <pvkernel/core/PVStringUtils.h>

#include <tbb/parallel_reduce.h>

#include <omp.h>


Picviz::PVMappingFilterStringDefault::PVMappingFilterStringDefault(PVCore::PVArgumentList const& args):
	PVMappingFilter(),
	_case_sensitive(true) // This will be changed by set_args anyway
{
	INIT_FILTER(PVMappingFilterStringDefault, args);
}

DEFAULT_ARGS_FILTER(Picviz::PVMappingFilterStringDefault)
{
	PVCore::PVArgumentList args;
	args[PVCore::PVArgumentKey("convert-lowercase", "Convert strings to lower case")].setValue<bool>(false);
	return args;
}

void Picviz::PVMappingFilterStringDefault::set_args(PVCore::PVArgumentList const& args)
{
	Picviz::PVMappingFilter::set_args(args);
	_case_sensitive = !args["convert-lowercase"].toBool();
}

Picviz::PVMappingFilter::decimal_storage_type* Picviz::PVMappingFilterStringDefault::operator()(PVRush::PVNraw::const_trans_nraw_table_line const& values)
{
	assert(_dest);
	assert(values.size() >= _dest_size);

	// First compute all the "string factors"
	
	const ssize_t size = values.size();
	const bool case_sensitive = _case_sensitive;
	// Looks like this can be fine optimised with hand made SSE/AVX optimisation
#pragma omp parallel for
	for (ssize_t i = 0; i < size; i++) {
		_dest[i].storage_as_uint() = (uint32_t) PVCore::PVStringUtils::compute_str_factor(values[i], case_sensitive);
	}

	return _dest;
}

IMPL_FILTER(Picviz::PVMappingFilterStringDefault)
