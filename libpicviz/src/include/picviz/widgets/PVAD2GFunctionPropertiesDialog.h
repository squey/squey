#ifndef PVAD2GFUNCTIONPROPERTIESDIALOG_H
#define PVAD2GFUNCTIONPROPERTIESDIALOG_H

#include <pvbase/export.h>
#include <picviz/widgets/PVAD2GFunctionPropertiesWidget.h>

#include <QDialog>

namespace Picviz {
class PVView;
}

namespace PVWidgets {

class LibPicvizDecl PVAD2GFunctionPropertiesDialog: public QDialog
{
public:
	PVAD2GFunctionPropertiesDialog(Picviz::PVView const& view_org, Picviz::PVView const& view_dst, Picviz::PVSelRowFilteringFunction const& rff, QWidget* parent = NULL);

public:
	inline Picviz::PVSelRowFilteringFunction_p get_rff() const { return _props->get_rff(); }

private:
	PVAD2GFunctionPropertiesWidget* _props;
};

}

#endif
