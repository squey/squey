#include <pvkernel/filter/PVFieldsFilterWidget.h>



PVFilter::PVFieldsFilterWidget::PVFieldsFilterWidget(PVFilter::fields_filter_type type, QString const& name_filter) :
	_name_filter(name_filter),
	_type(type)
{
	PVFieldsBaseFilter_p pf = LIB_FILTER(PVFilter::PVFieldsFilter<type>)::get()->get_filter_by_name(name_filter);
	assert(pf);
	_filter = pf->clone();
}

PVFilter::PVFieldsBaseFilter_p PVFilter::PVFieldsFilterWidget::get_filter_lib()
{
	return _filter;
}
