/**
 * \file PVFieldsFilterParamWidget.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVFILTER_PVFIELDSFILTERPARAMWIDGET_H
#define PVFILTER_PVFIELDSFILTERPARAMWIDGET_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVRegistrableClass.h>
#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/filter/PVFieldsFilter.h>

#include <QWidget>
#include <QObject>
#include <QString>
#include <QStringList>
#include <QObject>

namespace PVFilter {

class LibKernelDecl PVFieldsFilterParamWidgetBase : public QObject, public PVCore::PVRegistrableClass<PVFieldsFilterParamWidgetBase> {
	Q_OBJECT

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
    virtual PVCore::PVArgumentList get_default_argument() = 0;
	virtual size_t force_number_children() = 0;
	virtual void set_child_count(size_t count) = 0;
	virtual size_t get_child_count() = 0;
	virtual void clear_filter_data() = 0;
	virtual void push_data(QString const& data) = 0;
	virtual QStringList const& get_data() const = 0;
	virtual void update_data_display() = 0;

public:
	void connect_to_args_changed(QObject* dst, const char* slot)
	{
		connect(this, SIGNAL(args_changed_Signal()), dst, slot);
	}
	
	void connect_to_nchilds_changed(QObject* dst, const char* slot)
	{
		connect(this, SIGNAL(nchilds_changed_Signal()), dst, slot);
	}

signals:
	void args_changed_Signal();
	void nchilds_changed_Signal();
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
		_nchilds = 0;
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

	void update_data_display() { }

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

	virtual void clear_filter_data() { _filter_data.clear(); }
	virtual void push_data(QString const& data)
	{
		_filter_data << data;
	}

	virtual QStringList const& get_data() const { return _filter_data; }

    fields_filter_type type() {
        return _filter->type();
    }

    QString type_name() {
        return _filter->type_name();
    }

	virtual void set_child_count(size_t count) { _nchilds = count; }
	// That should be "type-specific" and returns 0 for a field filter !
	virtual size_t get_child_count() { return _nchilds; }

protected:
	virtual base_registrable* _clone_me() const { return NULL; }
protected:
    PVFilter::fields_filter_type _type;
    filter_p _filter;
	QStringList _filter_data;
	size_t _nchilds;
};

typedef PVFieldsFilterParamWidget<PVFilter::one_to_many> PVFieldsSplitterParamWidget;
typedef PVFieldsSplitterParamWidget::p_type PVFieldsSplitterParamWidget_p;

}

#ifdef WIN32
LibKernelDeclExplicitTempl PVCore::PVClassLibrary<PVFilter::PVFieldsSplitterParamWidget>;
LibKernelDeclExplicitTempl PVCore::PVTag<PVFilter::PVFieldsSplitterParamWidget>;
LibKernelDeclExplicitTempl PVCore::PVTag<PVFilter::PVFieldsFilterParamWidget<PVFilter::one_to_one> >;
LibKernelDeclExplicitTempl PVCore::PVTag<PVFilter::PVFieldsFilterParamWidget<PVFilter::many_to_many> >;

#endif

#endif
