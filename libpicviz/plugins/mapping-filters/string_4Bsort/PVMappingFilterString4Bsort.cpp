#include "PVMappingFilterString4Bsort.h"
#include <pvkernel/core/PVTBBMaxArray.h>

#include <tbb/parallel_reduce.h>

#include <omp.h>

inline float Picviz::PVMappingFilterString4Bsort::compute_str_factor(QString const& str)
{
	QByteArray value_as_qba = str.toUtf8();
	const char* value_as_char_p = value_as_qba.data();
	// int size = value_as_qba.size();

	// AG: what if my string has a length < 4 ?!
	char b1_c = value_as_char_p[0];
	char b2_c = value_as_char_p[1];
	char b3_c = value_as_char_p[2];
	char b4_c = value_as_char_p[3];

	float b1 = (float)b1_c;
	float b2 = (float)b2_c;
	float b3 = (float)b3_c;
	float b4 = (float)b4_c;

	float retval;

	retval = (((b4/256 + b3)/256 + b2)/256 +b1)/256;

	// PVLOG_INFO("Value:%f\n", retval);

	return retval;
}

float* Picviz::PVMappingFilterString4Bsort::operator()(PVRush::PVNraw::const_trans_nraw_table_line const& values)
{
	assert(_dest);
	assert(values.size() >= _dest_size);

	// First compute all the "string factors"
	int64_t size = _dest_size;

#pragma omp parallel for
	// Looks like this can be fine optimised with hand made SSE/AVX optimisation
	for (int64_t i = 0; i < size; i++) {
		_dest[i] = compute_str_factor(values[i]->get_qstr());
	}

	return _dest;
}

IMPL_FILTER_NOPARAM(Picviz::PVMappingFilterString4Bsort)
