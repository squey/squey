#ifndef PVFILTER_PVFIELDSFILTERPARAMWIDGET_H
#define PVFILTER_PVFIELDSFILTERPARAMWIDGET_H

#include <pvcore/general.h>
#include <pvcore/PVRegistrableClass.h>
#include <pvcore/PVClassLibrary.h>
#include <pvfilter/PVFieldsFilter.h>

#include <QWidget>
#include <QString>

namespace PVFilter {

class PVFieldsFilterParamWidgetBase: public PVCore::PVRegistrableClass<PVFieldsFilterParamWidgetBase>
{
public:
	typedef boost::shared_ptr< PVFieldsFilterParamWidgetBase > p_type;
public:
    virtual ~PVFieldsFilterParamWidgetBase() {}
public:
	virtual PVFilter::PVFieldsBaseFilter_p get_filter() = 0;
	virtual QWidget* get_param_widget() = 0;
    virtual QAction* get_action_menu() = 0;
    virtual void set_id(int id) = 0;
    virtual QString get_xml_tag() = 0;
};

typedef boost::shared_ptr<PVFieldsFilterParamWidgetBase> PVFieldsFilterParamWidgetBase_p;

template <PVFilter::fields_filter_type Ttype>
class PVFieldsFilterParamWidget: public PVFieldsFilterParamWidgetBase
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
    /**
     * @brief get the widget whiche is on the right of the GUI. It used to param the node.
     * @return widget ref
     */
	QWidget* get_param_widget() { return NULL; }
    /**
     * @brief get the action to push in menu
     * @return menu action
     */
    QAction* get_action_menu() { return NULL; }
    QString get_xml_tag() { return QString(""); }
    
    void set_id(int /*id*/) {}
    

	fields_filter_type type() { return _filter->type(); }
	QString type_name() { return _filter->type_name(); }
    

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
