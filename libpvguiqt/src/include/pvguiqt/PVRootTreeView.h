/**
 * \file PVRootTreeView.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVROOTTREEVIEW_H
#define PVROOTTREEVIEW_H

#include <pvkernel/core/general.h>
#include <QTreeView>

namespace PVCore {
class PVDataTreeObjectBase;
}

namespace Picviz {
class PVPlotted;
class PVMapped;
class PVView;
}

namespace PVGuiQt {

class PVRootTreeModel;

class PVRootTreeView: public QTreeView
{
	Q_OBJECT

public:
	PVRootTreeView(QAbstractItemModel* model, QWidget* parent = 0);

public:
	template <typename T, typename F>
	void visit_selected_objs_as(F const& f)
	{
		QModelIndexList sel = selectedIndexes();
		for (QModelIndex const& idx: sel) {
			PVCore::PVDataTreeObjectBase* obj_base = (PVCore::PVDataTreeObjectBase*) idx.internalPointer();
			T* obj = dynamic_cast<T*>(obj_base);
			if (obj) {
				f(obj);
			}
		}
	}

protected:
	void mouseDoubleClickEvent(QMouseEvent* event) override;
	void contextMenuEvent(QContextMenuEvent* event) override;
	void enterEvent(QEvent* event) override;
	void leaveEvent(QEvent* event) override;

protected slots:
	// Actions slots
	void create_new_view();
	void edit_mapping();
	void edit_plotting();

protected:
	PVCore::PVDataTreeObjectBase* get_selected_obj();

	template <typename T>
	T* get_selected_obj_as()
	{
		return dynamic_cast<T*>(this->get_selected_obj());
	}

protected:
	PVRootTreeModel* tree_model();
	PVRootTreeModel const* tree_model() const;

private:
	QAction* _act_new_view;
	QAction* _act_new_plotted;
	QAction* _act_new_mapped;
	QAction* _act_edit_mapping;
	QAction* _act_edit_plotting;

};

}

#endif // PVROOTTREEVIEW_H
