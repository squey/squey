//! \file PVMappingFilterIPv4Default.cpp
//! $Id$
//! Copyright (C) Sébastien Tricaud 2011-2012
//! Copyright (C) Philippe Saadé 2011-2012
//! Copyright (C) Picviz Labs 2012

#include "PVMappingFilterStringDefault.h"
#include <pvkernel/core/PVTBBMaxArray.h>
#include <pvkernel/core/PVCheckBoxType.h>
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
	//PVCore::PVCheckBoxType lowercase_checkbox;
	//lowercase_checkbox.set_checked(false);
	args[PVCore::PVArgumentKey("convert-lowercase", "Convert strings to lower case")].setValue<bool>(false);
	return args;
}

void Picviz::PVMappingFilterStringDefault::set_args(PVCore::PVArgumentList const& args)
{
	_case_sensitive = !args["convert-lowercase"].toBool();
	PVLOG_INFO("case sensitive: %d\n", _case_sensitive);
}

float* Picviz::PVMappingFilterStringDefault::operator()(PVRush::PVNraw::const_trans_nraw_table_line const& values)
{
	assert(_dest);
	assert(values.size() >= _dest_size);

	// First compute all the "string factors"
	
	int64_t size = _dest_size;
	float* factors = (float*) malloc(size * sizeof(float));
	if (factors == NULL) {
		PVLOG_ERROR("Unable to allocate a temporary array for string mapping !\n");
		return _dest;
	}

	const bool case_sensitive = _case_sensitive;
	PVLOG_INFO("\n\nPVMappingStringDefault case sensitive in operator(): %d\n", _case_sensitive);
#pragma omp parallel for
	// Looks like this can be fine optimised with hand made SSE/AVX optimisation
	for (int64_t i = 0; i < size; i++) {
		factors[i] = PVCore::PVStringUtils::compute_str_factor(values[i].get_qstr(), case_sensitive);
	}

	// Then find the maximum thanks to TBB
	PVCore::PVTBBMaxArray<float> mar(factors, STRING_MAX_YVAL);
	tbb::parallel_reduce(tbb::blocked_range<uint64_t>(0, size), mar, tbb::auto_partitioner());
	float max_factor = mar.get_max_value();

	// This is optimised by the compiler with SSE/AVX
	for (int64_t i = 0; i < size; i++) {
		_dest[i] = factors[i] / max_factor;
	}

	free(factors);

	return _dest;
}

IMPL_FILTER(Picviz::PVMappingFilterStringDefault)
