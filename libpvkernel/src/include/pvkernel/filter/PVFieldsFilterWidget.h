#ifndef PVFILTER_PVFIELDSFILTERWIDGET_H
#define PVFILTER_PVFIELDSFILTERWIDGET_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVRegistrableClass.h>
#include <pvkernel/core/PVClassLibrary.h>

namespace PVFilter {

class PVFieldsFilterWidget: public PVCore::PVRegistrableClass<PVFieldsFilterWidget>
{
public:
	typedef boost::shared_ptr<PVFieldsFilterWidget> p_type;
public:
	PVFieldsFilterWidget(PVFilter::fields_filter_type type, QString const& name_filter);

public:
	PVFilter::PVFieldsBaseFilter_p get_filter();

public:
	virtual QWidget* get_param_widget() = 0;

protected:
	QString _name_filter;
	PVFilter::fields_filter_type _type;
	PVFilter::PVFieldsBaseFilter_p _filter;
};

}
#ifdef WIN32
pvfilter_FilterLibraryDecl PVCore::PVClassLibrary<PVFilter::PVFieldsFilterWidget>;
#endif

#endif
