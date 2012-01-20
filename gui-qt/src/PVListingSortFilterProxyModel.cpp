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

bool PVInspector::PVListingSortFilterProxyModel::is_equal(const QModelIndex &left, const QModelIndex &right) const
{
	PVCore::PVUnicodeString const* sleft = (PVCore::PVUnicodeString const*) sourceModel()->data(left, PVCustomQtRoles::Sort).value<void*>();
	PVCore::PVUnicodeString const* sright = (PVCore::PVUnicodeString const*) sourceModel()->data(right, PVCustomQtRoles::Sort).value<void*>();
	return _equals_f(*sleft, *sright);
}

void PVInspector::PVListingSortFilterProxyModel::filter_source_indexes(vec_indexes_t const& src_idxes_in, vec_indexes_t& src_idxes_out)
{
	Picviz::PVSelection const* sel = _lib_view->get_selection_visible_listing();
	// If everything is displayed, just "copy" in to out.
	// A QVector is used, so no memory copy occurs.
	if (sel == NULL) {
		src_idxes_out = src_idxes_in;
		return;
	}

	// Filter out lines according to the good selection.
	src_idxes_out.clear();
	// AG: the allocation strategy isn't trivial. Indeed, in order to get the number
	// of visible lines, we need to check every bits of PVSelection (that's what's done in
	// get_number_of_selected_lines_in_range). There was an attempt to optimise this w/ SSE
	// and w/ noi branchment, but it gave false result.
	// Until this is not fixed, we will reserve for "src_idxes_out" the same number of lines than in
	// "src_idxes_in".
	//PVRow nvisible_lines = sel->get_number_of_selected_lines_in_range(0, _lib_view->get_row_count());
	src_idxes_out.reserve(src_idxes_in.size());
	vec_indexes_t::const_iterator it;
	for (it = src_idxes_in.begin(); it != src_idxes_in.end(); it++) {
		PVRow line = *it;
		if (sel->get_line(line)) {
			src_idxes_out.push_back(line);
		}
	}
}

void PVInspector::PVListingSortFilterProxyModel::sort_indexes(int column, Qt::SortOrder order, vec_indexes_t& vec_idxes)
{
	_lib_view->sort_indexes_with_axes_combination(column, order, vec_idxes);
}

void PVInspector::PVListingSortFilterProxyModel::sort(int column, Qt::SortOrder order)
{
	// TODO: get this from format, view, etc...
	_sort_f = _def_sort.f_less();
	_equals_f = _def_sort.f_equals();
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
