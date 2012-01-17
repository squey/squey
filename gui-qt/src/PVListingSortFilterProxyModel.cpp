#include <pvkernel/core/PVUnicodeString.h>
#include <picviz/PVView.h>

#include <PVListingSortFilterProxyModel.h>
#include <PVCustomQtRoles.h>

PVInspector::PVListingSortFilterProxyModel::PVListingSortFilterProxyModel(QObject* parent):
	QSortFilterProxyModel(parent)
{
	setDynamicSortFilter(true);
}

bool PVInspector::PVListingSortFilterProxyModel::filterAcceptsRow(int sourceRow, const QModelIndex& parent) const
{
	return true;
}

bool PVInspector::PVListingSortFilterProxyModel::lessThan(const QModelIndex &left, const QModelIndex &right) const
{
	PVCore::PVUnicodeString const* sleft = (PVCore::PVUnicodeString const*) sourceModel()->data(left, PVCustomQtRoles::Sort).value<void*>();
	PVCore::PVUnicodeString const* sright = (PVCore::PVUnicodeString const*) sourceModel()->data(right, PVCustomQtRoles::Sort).value<void*>();
	return _sort_f(*sleft, *sright);
}

void PVInspector::PVListingSortFilterProxyModel::sort(int column, Qt::SortOrder order)
{
	// TODO: get this from format, view, etc...
	_sort_f = _def_sort.f();
	QSortFilterProxyModel::sort(column, order);
}
