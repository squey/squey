#include <pvkernel/core/PVUnicodeString.h>
#include <picviz/PVStateMachine.h>
#include <picviz/PVView.h>

#include <PVListingSortFilterProxyModel.h>
#include <PVCustomQtRoles.h>
#include <PVTabSplitter.h>


PVInspector::PVListingSortFilterProxyModel::PVListingSortFilterProxyModel(PVTabSplitter* tab_parent, QObject* parent):
	PVSortFilterProxyModel(parent),
	_tab_parent(tab_parent)
{
	reset_lib_view();
}

bool PVInspector::PVListingSortFilterProxyModel::less_than(const QModelIndex &left, const QModelIndex &right) const
{
	PVCore::PVUnicodeString const* sleft = (PVCore::PVUnicodeString const*) sourceModel()->data(left, PVCustomQtRoles::Sort).value<void*>();
	PVCore::PVUnicodeString const* sright = (PVCore::PVUnicodeString const*) sourceModel()->data(right, PVCustomQtRoles::Sort).value<void*>();
	return _sort_f(*sleft, *sright);
}

bool PVInspector::PVListingSortFilterProxyModel::filter_source_index(int idx_in)
{
	return _lib_view->is_line_visible_listing(idx_in);
}

void PVInspector::PVListingSortFilterProxyModel::sort(int column, Qt::SortOrder order)
{
	// TODO: get this from format, view, etc...
	_sort_f = _def_sort.f();
	PVSortFilterProxyModel::sort(column, order);
}

void PVInspector::PVListingSortFilterProxyModel::refresh_filter()
{
	invalidate_filter();
}

void PVInspector::PVListingSortFilterProxyModel::reset_lib_view()
{
	_lib_view = _tab_parent->get_lib_view().get();
	assert(_lib_view);
	_state_machine = _lib_view->state_machine;
	invalidate_all();
}
