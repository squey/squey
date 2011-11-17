#include <picviz/PVPlottingFilter.h>
#include <pvkernel/core/stdint.h>

Picviz::PVPlottingFilter::PVPlottingFilter() :
	PVFilter::PVFilterFunctionBase<float*, float*>(),
	PVCore::PVRegistrableClass<Picviz::PVPlottingFilter>()
{
	_dest = NULL;
	_dest_size = 0;
	_mandatory_params = NULL;
}

float* Picviz::PVPlottingFilter::operator()(float* values)
{
	assert(values);
	assert(_dest);
	assert(_mandatory_params);

	int64_t size = _dest_size;
#pragma omp parallel for
	for (int64_t i = 0; i < size; i++) {
		_dest[i] = operator()(values[i]);
	}

	return _dest;
}

float Picviz::PVPlottingFilter::operator()(float /*value*/)
{
	PVLOG_WARN("In default plotting filter: returns 0 !\n");
	return 0;
}

void Picviz::PVPlottingFilter::set_mapping_mode(QString const& mode)
{
	_mapping_mode = mode;
}

void Picviz::PVPlottingFilter::set_dest_array(PVRow size, float* array)
{
	assert(array);
	_dest_size = size;
	_dest = array;
}

void Picviz::PVPlottingFilter::set_mandatory_params(Picviz::mandatory_param_map const& params)
{
	_mandatory_params = &params;
}

QStringList Picviz::PVPlottingFilter::list_modes(QString const& type)
{
	LIB_CLASS(PVPlottingFilter)::list_classes const& pl_filters = LIB_CLASS(PVPlottingFilter)::get().get_list();
	LIB_CLASS(PVPlottingFilter)::list_classes::const_iterator it;
	QStringList ret;
	for (it = pl_filters.begin(); it != pl_filters.end(); it++) {
		QString const& name = it.key();
		QStringList params = name.split('_');
		if (params[0].compare(type) == 0) {
			ret << params[1];
		}
	}
    return ret;
}

QString Picviz::PVPlottingFilter::get_human_name() const
{
	QStringList params = registered_name().split('_');
	return params[1];
}
