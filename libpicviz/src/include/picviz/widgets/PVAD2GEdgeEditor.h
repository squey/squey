#ifndef __PVAD2GEDGEEDITOR_H__
#define __PVAD2GEDGEEDITOR_H__

#include <QDialog>
#include <QListView>

#include <pvkernel/core/general.h>
#include <picviz/PVSelRowFilteringFunction_types.h>
#include <picviz/widgets/PVAD2GRFFListModel.h>

namespace Picviz {
class PVView;
class PVCombiningFunctionView;
class PVTFViewRowFiltering;
}

namespace PVWidgets {

class LibPicvizDecl PVAD2GEdgeEditor : public QDialog
{
	Q_OBJECT
public:
	PVAD2GEdgeEditor(Picviz::PVView const& view_org, Picviz::PVView const& view_dst, Picviz::PVCombiningFunctionView& cf, QWidget* parent = 0);

public slots:
	void add_function_Slot();
	void edit_function_Slot();
	void remove_function_Slot();

private:
	bool edit_rff(Picviz::PVSelRowFilteringFunction_p& rff);

private:
	static Picviz::PVSelRowFilteringFunction_p get_default_rff();

private:
	PVAD2GRFFListModel* _rff_list_model;
	QListView* _list;

	Picviz::PVTFViewRowFiltering& _tf;
	Picviz::PVView const& _view_org;
	Picviz::PVView const& _view_dst;
};

}

#endif // __PVAD2GEDGEEDITOR_H__
