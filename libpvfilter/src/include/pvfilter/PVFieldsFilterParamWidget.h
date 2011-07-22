#ifndef PVFILTER_PVFIELDSFILTERPARAMWIDGET_H
#define PVFILTER_PVFIELDSFILTERPARAMWIDGET_H

#include <pvcore/general.h>
#include <pvcore/PVRegistrableClass.h>
#include <pvcore/PVClassLibrary.h>
#include <pvfilter/PVFieldsFilter.h>

#include <QWidget>
#include <QString>
#include <QObject>

namespace PVFilter {

class PVFieldsFilterParamWidgetBase : public PVCore::PVRegistrableClass<PVFieldsFilterParamWidgetBase> {
public:
    typedef boost::shared_ptr< PVFieldsFilterParamWidgetBase > p_type;
public:

    virtual ~PVFieldsFilterParamWidgetBase() {
    }
public:
    virtual PVFilter::PVFieldsBaseFilter_p get_filter() = 0;
    virtual QWidget* get_param_widget() = 0;
    virtual QAction* get_action_menu() = 0;
    virtual void set_id(int id) = 0;
    virtual QString get_xml_tag() = 0;
    virtual QObject* get_as_qobject() = 0;
    virtual PVCore::PVArgumentList get_default_argument() = 0;
    virtual void set_child_count(int count) = 0;
    virtual int get_child_new_num() = 0;
	virtual size_t force_number_children() = 0;
};

typedef boost::shared_ptr<PVFieldsFilterParamWidgetBase> PVFieldsFilterParamWidgetBase_p;

template <PVFilter::fields_filter_type Ttype>
class PVFieldsFilterParamWidget : public PVFieldsFilterParamWidgetBase {
public:
    typedef PVFieldsFilterParamWidget<Ttype> RegAs;
    typedef boost::shared_ptr< PVFieldsFilterParamWidget<Ttype> > p_type;
	typedef typename PVFilter::PVFieldsFilter<Ttype>::p_type filter_p;
public:

    PVFieldsFilterParamWidget(filter_p filter) {
        assert(filter);
        _filter = filter->clone< PVFieldsFilter<Ttype> >();
    }

    PVCore::PVArgumentList get_default_argument(){
		return _filter->get_default_args();
    }

    PVFieldsBaseFilter_p get_filter() {
        return _filter;
    }

    /**
     * @brief get the widget whiche is on the right of the GUI. It used to param the node.
     * @return widget ref
     */
    QWidget* get_param_widget() {
        return NULL;
    }

    QObject* get_as_qobject() {
        return NULL;
    }

	// Force the number of children. Returns 0 if no forcing is done.
	// TODO: this should only exist when Ttype == one_to_many, and should not be in the base interface
	// (but that's for the "purity" of C++, and, well, don't have the time for this right now)
	size_t force_number_children() {
		return 0;
	}

    /**
     * @brief get the action to push in menu
     * @return menu action
     */
    QAction* get_action_menu() {
        return NULL;
    }

    QString get_xml_tag() {
		return type_name();
    }

    void set_id(int /*id*/) {
    }
    
    void set_child_count(int ) {}
    
    int get_child_new_num() {return 0;}

    fields_filter_type type() {
        return _filter->type();
    }

    QString type_name() {
        return _filter->type_name();
    }

protected:
    PVFilter::fields_filter_type _type;
    filter_p _filter;
};

typedef PVFieldsFilterParamWidget<PVFilter::one_to_many> PVFieldsSplitterParamWidget;
typedef PVFieldsSplitterParamWidget::p_type PVFieldsSplitterParamWidget_p;

}

#ifdef WIN32
pvfilter_FilterLibraryDecl PVCore::PVClassLibrary<PVFilter::PVFieldsSplitterParamWidget>;
#endif

#endif
