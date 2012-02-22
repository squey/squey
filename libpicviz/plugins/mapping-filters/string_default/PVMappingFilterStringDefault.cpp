//! \file PVMappingFilterIPv4Default.cpp
//! $Id$
//! Copyright (C) Sébastien Tricaud 2011-2012
//! Copyright (C) Philippe Saadé 2011-2012
//! Copyright (C) Picviz Labs 2012

#include "PVMappingFilterStringDefault.h"
#include <pvkernel/core/PVTBBMaxArray.h>
#include <pvkernel/core/PVCheckBoxType.h>
#include <pvkernel/core/string.h>

#include <tbb/parallel_reduce.h>

#include <omp.h>


Picviz::PVMappingFilterStringDefault::PVMappingFilterStringDefault(PVCore::PVArgumentList const& args):
	PVMappingFilter()
{
	INIT_FILTER(PVMappingFilterStringDefault, args);
}

DEFAULT_ARGS_FILTER(Picviz::PVMappingFilterStringDefault)
{
	PVCore::PVArgumentList args;
	PVCore::PVCheckBoxType lowercase_checkbox;
	args[PVCore::PVArgumentKey("convert-lowercase", "Convert strings to lower case")].setValue(lowercase_checkbox);
	return args;
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

#pragma omp parallel for
	// Looks like this can be fine optimised with hand made SSE/AVX optimisation
	for (int64_t i = 0; i < size; i++) {
		factors[i] = PVCore::String::compute_str_factor(values[i].get_qstr());
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

IMPL_FILTER_NOPARAM(Picviz::PVMappingFilterStringDefault)
