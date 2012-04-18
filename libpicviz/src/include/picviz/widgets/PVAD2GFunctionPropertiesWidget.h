#ifndef PVAD2GFUNCTIONPROPERTIES_WIDGET_H
#define PVAD2GFUNCTIONPROPERTIES_WIDGET_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVArgument.h>
#include <picviz/PVSelRowFilteringFunction.h>

#include <QWidget>

namespace PVWidgets {
class PVArgumentListWidget;
}

namespace Picviz {

class PVView;

class LibPicvizDecl PVAD2GFunctionPropertiesWidget: public QWidget
{
	Q_OBJECT
	
	typedef std::map<PVSelRowFilteringFunction, PVCore::PVArgumentList> map_rff_args_t;
public:
	PVAD2GFunctionPropertiesWidget(PVView const& view_org, PVView const& view_dst, PVSelRowFilteringFunction_p rff_p, QWidget* parent = NULL);

public:
	PVSelRowFilteringFunction_p get_rff() const;

protected:
	void set_current_rff(PVSelRowFilteringFunction_p const& rff);

protected:
	// Constant properties (views)
	PVView const& _view_org;
	PVView const& _view_dst;

	// RFFs objects
	PVSelRowFilteringFunction_p _cur_rff;
	map_rff_args_t _rffs_args;

	// Arguments for different views
	PVCore::PVArgumentList _args_org;
	PVCore::PVArgumentList _args_dst;
	PVWidgets::PVArgumentListWidget* _args_org_widget;
	PVWidgets::PVArgumentListWidget* _args_dst_widget;
};

}

#endif
