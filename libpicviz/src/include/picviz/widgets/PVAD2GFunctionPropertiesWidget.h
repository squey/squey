/**
 * \file PVAD2GFunctionPropertiesWidget.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVAD2GFUNCTIONPROPERTIES_WIDGET_H
#define PVAD2GFUNCTIONPROPERTIES_WIDGET_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVArgument.h>
#include <picviz/PVSelRowFilteringFunction.h>

#include <QComboBox>
#include <QGroupBox>
#include <QWidget>
#include <QStackedWidget>

namespace Picviz {
class PVView;
}

namespace PVWidgets {

class PVArgumentListWidget;

class LibPicvizDecl PVAD2GFunctionPropertiesWidget: public QWidget
{
	Q_OBJECT
	
	typedef std::map<Picviz::PVSelRowFilteringFunction::base_registrable, PVCore::PVArgumentList> map_rff_args_t;
public:
	PVAD2GFunctionPropertiesWidget(QWidget* parent = NULL);

public:
	void set_views(Picviz::PVView const& view_org, Picviz::PVView const& view_dst);
	void set_current_rff(Picviz::PVSelRowFilteringFunction const* rff, bool keep_args = true);
	Picviz::PVSelRowFilteringFunction_p get_rff() const { return _cur_rff; };

signals:
	void function_properties_changed(const Picviz::PVSelRowFilteringFunction_p& rff);

public slots:
	void commit_args();

protected:
	void init_combo_list_rffs();
	static void set_args_widget(PVArgumentListWidget* widgets, PVCore::PVArgumentList& args);

protected slots:
	void combo_func_changed(int idx);

protected:
	// Constant properties (views)
	const Picviz::PVView* _view_org;
	const Picviz::PVView* _view_dst;

	PVWidgets::PVArgumentListWidget* _args_org_widget;
	PVWidgets::PVArgumentListWidget* _args_dst_widget;
	PVWidgets::PVArgumentListWidget* _args_global_widget;

	// RFFs objects
	Picviz::PVSelRowFilteringFunction_p _cur_rff;
	map_rff_args_t _rffs_args;

	// Arguments for different views
	PVCore::PVArgumentList _args_org;
	PVCore::PVArgumentList _args_dst;
	PVCore::PVArgumentList _args_global;

	// Widgets
	QComboBox* _function_combo;
	QGroupBox* _global_view_box;
	QGroupBox* _src_view_box;
	QGroupBox* _dst_view_box;

	QStackedWidget* _stacked_widget;
	QWidget* _widget;
};

}

#endif
