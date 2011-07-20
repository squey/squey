#ifndef PVFILTER_PVFIELDSFILTERPARAMWIDGET_H
#define PVFILTER_PVFIELDSFILTERPARAMWIDGET_H

#include <pvcore/general.h>
#include <pvcore/PVRegistrableClass.h>
#include <pvcore/PVClassLibrary.h>
#include <pvfilter/PVFieldsFilter.h>

#include <QWidget>

namespace PVFilter {

class LibFilterDecl PVFieldsFilterParamWidgetBase: public PVCore::PVRegistrableClass<PVFieldsFilterParamWidgetBase>
{
public:
	typedef boost::shared_ptr< PVFieldsFilterParamWidgetBase > p_type;
public:
	virtual PVFilter::PVFieldsBaseFilter_p get_filter() = 0;
	virtual QWidget* get_param_widget() = 0;
};

template <PVFilter::fields_filter_type Ttype>
class LibFilterDecl PVFieldsFilterParamWidget: public PVFieldsFilterParamWidgetBase
{
public:
	typedef PVFieldsFilterParamWidget<Ttype> RegAs;
	typedef boost::shared_ptr< PVFieldsFilterParamWidget<Ttype> > p_type;
public:
	PVFieldsFilterParamWidget(typename PVFilter::PVFieldsFilter<Ttype>::p_type filter)
	{
		assert(filter);
		_filter = filter->clone< PVFieldsFilter<Ttype> >();
	}

	PVFieldsBaseFilter_p get_filter() { return _filter; }
	QWidget* get_param_widget() { return NULL; }

protected:
	QString _name_filter;
	PVFilter::fields_filter_type _type;
	typename PVFilter::PVFieldsFilter<Ttype>::p_type _filter;
};

typedef PVFieldsFilterParamWidget<PVFilter::one_to_many> PVFieldsSplitterParamWidget;
typedef PVFieldsSplitterParamWidget::p_type PVFieldsSplitterParamWidget_p;

}

#ifdef WIN32
pvfilter_FilterLibraryDecl PVCore::PVClassLibrary<PVFilter::PVFieldsSplitterParamWidget>;
#endif

#endif
