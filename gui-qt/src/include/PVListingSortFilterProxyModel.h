#ifndef PVINSPECTOR_PVLISTINGFILTERPROXYMODEL_H
#define PVINSPECTOR_PVLISTINGFILTERPROXYMODEL_H

#include <picviz/PVDefaultSortingFunc.h>
#include <picviz/PVSortingFunc.h>
#include <picviz/PVView_types.h>

#include <QSortFilterProxyModel>

namespace Picviz {
class PVStateMachine;
}

namespace PVInspector {

class PVTabSplitter;

class PVListingSortFilterProxyModel: public QSortFilterProxyModel
{
public:
	PVListingSortFilterProxyModel(PVTabSplitter* tab_parent, QObject* parent = NULL);

public:
	void refresh_filter();
	void reset_lib_view();

protected:
	bool filterAcceptsRow(int row, const QModelIndex &parent) const;
	bool lessThan(const QModelIndex &left, const QModelIndex &right) const;
	void sort(int column, Qt::SortOrder order);

private:
	mutable Picviz::PVSortingFunc_f _sort_f;
	Picviz::PVView* _lib_view;
	Picviz::PVStateMachine* _state_machine;
	PVTabSplitter* _tab_parent;

	// Temporary
	Picviz::PVDefaultSortingFunc _def_sort;

	Q_OBJECT
};

}

#endif
