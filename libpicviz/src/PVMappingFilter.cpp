#include <picviz/PVMappingFilter.h>
#include <pvrush/PVFormat.h>

#include <pvcore/stdint.h>

Picviz::PVMappingFilter::PVMappingFilter()
{
	_dest = NULL;
	_format = NULL;
	_cur_col = 0;
}

float* Picviz::PVMappingFilter::operator()(PVRush::PVNraw::nraw_table_line const& values)
{
	assert(_dest);
	assert(values.size() >= _dest_size);

	init_from_first(values[0]);
#pragma omp parallel for
	for (int64_t i = 0; i < _dest_size; i++) {
		_dest[i] = operator()(values[i]);
	}

	return _dest;
}

float Picviz::PVMappingFilter::operator()(QString const& /*value*/)
{
	PVLOG_WARN("In default mapping filter: does nothing !\n");
	return 0;
}

void Picviz::PVMappingFilter::init_from_first(QString const& /*value*/)
{
}

void Picviz::PVMappingFilter::set_dest_array(PVRow size, float* ptr)
{
	assert(ptr);
	// This array is supposed to be as large as the values given to operator()
	_dest = ptr;
	_dest_size = size;
}

void Picviz::PVMappingFilter::set_format(PVCol cur_col, PVRush::PVFormat& format)
{
	_cur_col = cur_col;
	_format = &format;
}
