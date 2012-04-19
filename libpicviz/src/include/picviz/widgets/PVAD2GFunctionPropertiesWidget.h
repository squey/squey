#ifndef PVAD2GFUNCTIONPROPERTIES_WIDGET_H
#define PVAD2GFUNCTIONPROPERTIES_WIDGET_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVArgument.h>
#include <picviz/PVSelRowFilteringFunction.h>

#include <QComboBox>
#include <QGroupBox>
#include <QWidget>

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
	PVAD2GFunctionPropertiesWidget(Picviz::PVView const& view_org, Picviz::PVView const& view_dst, Picviz::PVSelRowFilteringFunction const& rff, QWidget* parent = NULL);

public:
	Picviz::PVSelRowFilteringFunction_p get_rff() const { return _cur_rff; };

public slots:
	void commit_args();

protected:
	void init_combo_list_rffs(Picviz::PVSelRowFilteringFunction const* rff);
	void set_current_rff(Picviz::PVSelRowFilteringFunction const* rff);

protected slots:
	void combo_func_changed(int idx);

protected:
	// Constant properties (views)
	Picviz::PVView const& _view_org;
	Picviz::PVView const& _view_dst;

	// RFFs objects
	Picviz::PVSelRowFilteringFunction_p _cur_rff;
	map_rff_args_t _rffs_args;

	// Arguments for different views
	PVCore::PVArgumentList _args_org;
	PVCore::PVArgumentList _args_dst;
	PVCore::PVArgumentList _args_global;

	PVWidgets::PVArgumentListWidget* _args_org_widget;
	PVWidgets::PVArgumentListWidget* _args_dst_widget;
	PVWidgets::PVArgumentListWidget* _args_global_widget;

	// Widgets
	QComboBox* _function_combo;
	QGroupBox* _global_view_box;
	QGroupBox* _src_view_box;
	QGroupBox* _dst_view_box;
};

}

#endif
