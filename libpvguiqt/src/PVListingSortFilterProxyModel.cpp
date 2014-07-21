/**
 * \file PVListingSortFilterProxyModel.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#define TBB_PREVIEW_DETERMINISTIC_REDUCE 1
#include <tbb/parallel_reduce.h>
#include <tbb/task_scheduler_init.h>

#include <pvkernel/core/PVUnicodeString.h>
#include <pvkernel/core/PVHardwareConcurrency.h>

#include <picviz/PVView.h>
#include <picviz/PVSortingFunc.h>

#include <pvguiqt/PVListingSortFilterProxyModel.h>
#include <pvguiqt/PVCustomQtRoles.h>

#if 0
template <typename T>
class ParallelFilterIndexes
{
public:
	typedef PVGuiQt::PVSortFilterProxyModel::vec_indexes_t buffer_t;
	typedef tbb::blocked_range<size_t> blocked_range;

public:
	ParallelFilterIndexes() : _count(0)
	{}

	ParallelFilterIndexes(const buffer_t &src_idxes_in,
	                      buffer_t &src_idxes_out,
	                      Picviz::PVSelection const* sel) :
		_offset(0), _count(0),
		_in(src_idxes_in), _out(src_idxes_out), _sel(sel)
	{}

	ParallelFilterIndexes(ParallelFilterIndexes& s, tbb::split) :
		_offset(0), _count(0),
		_in(s._in), _out(s._out), _sel(s._sel)
	{}

	size_t get_count()
	{
		return _count;
	}

	void operator()(const blocked_range &r)
	{
		size_t offset = _offset = r.begin();
		for(size_t i = r.begin(); i != r.end(); ++i) {
			const PVRow line = _in[i];
			if (_sel->get_line(line)) {
				_out[offset] = line;
				++_count;
				++offset;
			}
		}
	}

	void join(ParallelFilterIndexes &rhs)
	{
		memmove(_out.get() + _offset + _count,
		        _out.get() + rhs._offset,
		        sizeof(T) * rhs._count);
		_count += rhs._count;
	}

private:
	size_t                     _offset;
	size_t                     _count;
	const buffer_t            &_in;
	buffer_t                  &_out;
	const Picviz::PVSelection *_sel;

};
#endif

void PVGuiQt::__impl::PVListingVisibilityObserver::update(arguments_type const&) const
{
	_parent->refresh_filter();
}

void PVGuiQt::__impl::PVListingVisibilityZombieObserver::update(arguments_type const&) const
{
	_parent->refresh_filter();
}

PVGuiQt::PVListingSortFilterProxyModel::PVListingSortFilterProxyModel(Picviz::PVView_sp& lib_view, QTableView* view, QObject* parent):
	PVSortFilterProxyModel(view, parent),
	_lib_view(*lib_view.get()),
	_obs_vis(this),
	_obs_zomb(this)
{
	invalidate_all();

	_obs_output_layer.connect_refresh(this, SLOT(refresh_filter()));
	_obs_sel.connect_refresh(this, SLOT(refresh_filter()));
	PVHive::get().register_observer(lib_view, [=](Picviz::PVView& view) { return &view.get_output_layer(); }, _obs_output_layer);
	PVHive::get().register_observer(lib_view, [=](Picviz::PVView& view) { return &view.get_real_output_selection(); }, _obs_sel);
	PVHive::get().register_func_observer(lib_view, _obs_vis);
	PVHive::get().register_func_observer(lib_view, _obs_zomb);
}

bool PVGuiQt::PVListingSortFilterProxyModel::less_than(const QModelIndex &left, const QModelIndex &right) const
{
	PVCore::PVUnicodeString const* sleft = (PVCore::PVUnicodeString const*) sourceModel()->data(left, PVCustomQtRoles::Sort).value<void*>();
	PVCore::PVUnicodeString const* sright = (PVCore::PVUnicodeString const*) sourceModel()->data(right, PVCustomQtRoles::Sort).value<void*>();
	return _sort_f(*sleft, *sright);
}

bool PVGuiQt::PVListingSortFilterProxyModel::is_equal(const QModelIndex &left, const QModelIndex &right) const
{
	const std::string& strleft = sourceModel()->data(left, PVCustomQtRoles::Sort).value<std::string>();
	const std::string& strright = sourceModel()->data(right, PVCustomQtRoles::Sort).value<std::string>();
	PVCore::PVUnicodeString sleft(strleft.c_str(), strleft.length());
	PVCore::PVUnicodeString sright(strright.c_str(), strright.length());
	return _equals_f(sleft, sright);
}

void PVGuiQt::PVListingSortFilterProxyModel::filter_source_indexes(vec_indexes_t const& src_idxes_in, vec_indexes_t& src_idxes_out)
{
	Picviz::PVSelection const* sel = _lib_view.get_selection_visible_listing();
	// If everything is displayed, just "copy" in to out.
	// A QVector is used, so no memory copy occurs.
	if (sel == NULL) {
		src_idxes_out = src_idxes_in;
		return;
	}

	// Filter out lines according to the good selection.
	src_idxes_out.clear();
	const PVRow nvisible_lines = sel->get_number_of_selected_lines_in_range(0, _lib_view.get_row_count());
	if (nvisible_lines == 0) {
		return;
	}

	src_idxes_out.reserve(nvisible_lines);
	vec_indexes_t::const_iterator it;
	for (it = src_idxes_in.begin(); it != src_idxes_in.end(); it++) {
		PVRow line = *it;
		if (sel->get_line(line)) {
			src_idxes_out.push_back(line);
		}
	}
}

void PVGuiQt::PVListingSortFilterProxyModel::sort_indexes(int column, Qt::SortOrder order, vec_indexes_t& vec_idxes, tbb::task_group_context* ctxt /*= NULL*/)
{
	_lib_view.sort_indexes_with_axes_combination(column, order, vec_idxes, ctxt);
}

void PVGuiQt::PVListingSortFilterProxyModel::sort(int column, Qt::SortOrder order)
{
	Picviz::PVSortingFunc_p sf = _lib_view.get_sort_plugin_for_col(column);
	_sort_f = sf->f_lesser();
	_equals_f = sf->f_equals();
	PVSortFilterProxyModel::sort(column, order);
}

void PVGuiQt::PVListingSortFilterProxyModel::refresh_filter()
{
	invalidate_filter();
}
