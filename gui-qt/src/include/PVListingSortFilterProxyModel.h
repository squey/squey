#ifndef PVINSPECTOR_PVLISTINGFILTERPROXYMODEL_H
#define PVINSPECTOR_PVLISTINGFILTERPROXYMODEL_H

#include <picviz/PVDefaultSortingFunc.h>
#include <picviz/PVSortingFunc.h>

#include <QSortFilterProxyModel>

namespace PVInspector {

class PVListingSortFilterProxyModel: public QSortFilterProxyModel
{
public:
	PVListingSortFilterProxyModel(QObject* parent = NULL);

protected:
	bool filterAcceptsRow(int row, const QModelIndex &parent) const;
	bool lessThan(const QModelIndex &left, const QModelIndex &right) const;
	void sort(int column, Qt::SortOrder order);

private:
	mutable Picviz::PVSortingFunc_f _sort_f;

	// Temporary
	Picviz::PVDefaultSortingFunc _def_sort;

	Q_OBJECT
};

}

#endif
