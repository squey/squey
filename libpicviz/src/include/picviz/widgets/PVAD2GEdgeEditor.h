#ifndef __PVAD2GEDGEEDITOR_H__
#define __PVAD2GEDGEEDITOR_H__

#include <QDialog>
#include <QListView>
#include <QComboBox>

#include <pvkernel/core/general.h>
#include <pvkernel/widgets/PVSizeHintListWidget.h>
#include <picviz/PVSelRowFilteringFunction_types.h>
#include <picviz/widgets/PVAD2GRFFListModel.h>


namespace Picviz {
class PVView;
class PVCombiningFunctionView;
class PVTFViewRowFiltering;
}

namespace PVWidgets {

class LibPicvizDecl PVAD2GEdgeEditor : public QWidget
{
	Q_OBJECT
public:
	PVAD2GEdgeEditor(Picviz::PVView const& view_org, Picviz::PVView const& view_dst, Picviz::PVCombiningFunctionView& cf, QWidget* parent = 0);
	PVAD2GEdgeEditor(QWidget* parent = 0);

public:
	void set_cf(Picviz::PVView const& view_org, Picviz::PVView const& view_dst, Picviz::PVCombiningFunctionView& cf);
	Picviz::PVView const* get_view_org() const { return _view_org; }
	Picviz::PVView const* get_view_dst() const { return _view_dst; }

public slots:
	void add_function_Slot();
	void edit_function_Slot();
	void remove_function_Slot();
	void update_item_Slot(const Picviz::PVSelRowFilteringFunction_p& rff);

signals:
	void update_fonction_properties(const Picviz::PVView& view_org, const Picviz::PVView& view_dst, Picviz::PVSelRowFilteringFunction_p& rff);
	void rff_list_changed();
	void cur_rff_removed();

private:
	void init();
	bool edit_rff(Picviz::PVSelRowFilteringFunction_p& rff);
	void init_combo_list_rffs();

private:
	PVAD2GRFFListModel* _rff_list_model;
	const Picviz::PVView* _view_org;
	const Picviz::PVView* _view_dst;
	QModelIndex _cur_edited_rff_index;

	PVSizeHintListWidget<QListView>* _list;
	QComboBox* _function_combo;

	/*Picviz::PVTFViewRowFiltering& _tf;*/

};

}

#endif // __PVAD2GEDGEEDITOR_H__
