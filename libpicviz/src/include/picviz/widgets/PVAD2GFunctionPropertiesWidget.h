#ifndef PVAD2GFUNCTIONPROPERTIES_WIDGET_H
#define PVAD2GFUNCTIONPROPERTIES_WIDGET_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVArgument.h>
#include <picviz/PVSelRowFilteringFunction.h>

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

protected:
	void set_current_rff(Picviz::PVSelRowFilteringFunction const* rff);

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
	PVWidgets::PVArgumentListWidget* _args_org_widget;
	PVWidgets::PVArgumentListWidget* _args_dst_widget;
};

}

#endif
