#include "PVMappingFilterStringDefault.h"
#include <pvcore/PVTBBMaxArray.h>
#include <pvcore/string.h>

#include <tbb/parallel_reduce.h>

#include <omp.h>


float* Picviz::PVMappingFilterStringDefault::operator()(PVRush::PVNraw::nraw_table_line const& values)
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
		factors[i] = PVCore::String::compute_str_factor(values[i]);
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
