#include <picviz/PVMappingFilter.h>
#include <pvkernel/rush/PVFormat.h>

#include <pvkernel/core/stdint.h>

Picviz::PVMappingFilter::PVMappingFilter()
{
	_dest = NULL;
	_format = NULL;
	_cur_col = 0;
	_grp_value = NULL;
}

float* Picviz::PVMappingFilter::operator()(PVRush::PVNraw::const_trans_nraw_table_line const& values)
{
	assert(_dest);
	assert(values.size() >= _dest_size);

	init_from_first(values[0].get_qstr());
#pragma omp parallel for
	for (int64_t i = 0; i < _dest_size; i++) {
		_dest[i] = operator()(values[i].get_qstr());
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

QStringList Picviz::PVMappingFilter::list_types()
{
	LIB_CLASS(PVMappingFilter)::list_classes const& map_filters = LIB_CLASS(PVMappingFilter)::get().get_list();
	LIB_CLASS(PVMappingFilter)::list_classes::const_iterator it;
	QStringList ret;
	for (it = map_filters.begin(); it != map_filters.end(); it++) {
		QString const& name = it.key();
		QStringList params = name.split('_');
		if (!ret.contains(params[0])) {
			ret << params[0];
		}
	}
    return ret;
}

QStringList Picviz::PVMappingFilter::list_modes(QString const& type)
{
	LIB_CLASS(Picviz::PVMappingFilter)::list_classes const& map_filters = LIB_CLASS(Picviz::PVMappingFilter)::get().get_list();
	LIB_CLASS(Picviz::PVMappingFilter)::list_classes::const_iterator it;
	QStringList ret;
	for (it = map_filters.begin(); it != map_filters.end(); it++) {
		QString const& name = it.key();
		QStringList params = name.split('_');
		if (params[0].compare(type) == 0) {
			ret << params[1];
		}
	}
    return ret;
}

QString Picviz::PVMappingFilter::get_human_name() const
{
	QStringList params = registered_name().split('_');
	return params[1];
}
