//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <squey/PVMapped.h>             // for PVMapped
#include <squey/PVMappingProperties.h>  // for PVMappingProperties
#include <squey/PVScaled.h>            // for PVScaled, etc
#include <squey/PVScalingFilter.h>     // for PVScalingFilter, etc
#include <squey/PVScalingProperties.h> // for PVScalingProperties
#include <squey/PVSelection.h>          // for PVSelection
#include <squey/PVSource.h>             // for PVSource
#include <squey/PVView.h>               // for PVView

#include <pvkernel/rush/PVFormat.h> // for PVFormat

#include <pvcop/db/algo.h> // for minmax

#include <pvkernel/core/PVColumnIndexes.h>   // for PVColumnIndexes
#include <pvkernel/core/PVDataTreeObject.h>  // for PVDataTreeChild
#include <pvkernel/core/PVLogger.h>          // for PVLOG_DEBUG
#include <pvkernel/core/PVSerializeObject.h> // for PVSerializeObject_p, etc

#include <pvbase/types.h> // for PVCol, PVRow

#include <boost/thread/thread.hpp>

#include <omp.h>

#include <QList>   // for QList
#include <QString> // for QString

#include <algorithm>  // for move, all_of
#include <cassert>    // for assert
#include <cstddef>    // for size_t
#include <cstdint>    // for uint32_t
#include <functional> // for _Mem_fn, mem_fn
#include <list>       // for _List_const_iterator, list
#include <memory>     // for allocator, __shared_ptr
#include <string>     // for string, operator+, etc
#include <vector>     // for vector

Squey::PVScaled::PVScaled(PVMapped& mapped, std::string const& name)
    : PVCore::PVDataTreeChild<PVMapped, PVScaled>(mapped), _name(name)
{
	PVRush::PVFormat const& format = get_parent<Squey::PVSource>().get_format();

	for (PVCol i(0); i < format.get_axes().size(); i++) {
		_columns.emplace_back(format, i);
	}
	create_table();
}

Squey::PVScaled::PVScaled(PVMapped& mapped,
                             std::list<Squey::PVScalingProperties>&& column,
                             std::string const& name)
    : PVCore::PVDataTreeChild<PVMapped, PVScaled>(mapped), _columns(std::move(column)), _name(name)
{
	create_table();
}

Squey::PVScaled::~PVScaled()
{
	PVLOG_DEBUG("In PVScaled destructor\n");
}

int Squey::PVScaled::create_table()
{
	const PVCol mapped_col_count = get_nraw_column_count();

	// Allocated once, then recomputed in place. Starting from zero here pushed a
	// second full set of columns on every later pass -- one array of a uint32 per
	// row each, none of them ever read, since every access indexes by column --
	// which selection scaling would repeat at every change of selection.
	for (size_t i = _scaleds.size(); i < _columns.size(); i++) {
		_scaleds.emplace_back(Squey::scaling_type, get_row_count());
	}

	_last_updated_cols.clear();
	_minmax_values.resize(mapped_col_count);
	_scaled_on_domain.resize(mapped_col_count, 0);
	_domain_bounds.resize(mapped_col_count);
	_domain_only_invalidation.resize(mapped_col_count, false);

	auto const& axes_format = get_parent<Squey::PVSource>().get_format().get_axes();

	// Which columns need a pass, settled first and on this thread alone: a column
	// whose scaling mode does not fit its type is given the default one here, and
	// what is left to do depends on that.
	std::vector<PVCol> to_scale;
	for (PVCol j(0); j < mapped_col_count; j++) {
		auto const& mapping_mode = get_parent<PVMapped>().get_properties_for_col(j).get_mode();
		auto usable_pt = get_properties_for_col(j).get_scaling_filter()->list_usable_type();
		if (not usable_pt.empty() and
		    usable_pt.count(
		        std::make_pair(axes_format[j].get_type().toStdString(), mapping_mode)) == 0) {
			get_properties_for_col(j).set_mode("default");
		}

		if (not get_properties_for_col(j).is_uptodate()) {
			to_scale.push_back(j);
		}
	}

	// One column at a time, each one parallel inside. Scaling them side by side
	// instead was measured slower: nested OpenMP is off, so the filters would run
	// serially within their own column, and with fewer columns than cores that
	// loses more than the outer parallelism wins.
	for (PVCol j : to_scale) {
		boost::this_thread::interruption_point();

		if (scale_column(j)) {
			_last_updated_cols.push_back(j);
		}
	}

	// Only the pass that follows refresh_selection_scaling() may skip a column on
	// the grounds that its bounds are unchanged. Left standing, this would answer
	// for later passes too -- a scaling mode changed from the axis menu, a mapping
	// recomputed -- and those have every reason to rewrite a column whose bounds
	// happen to be the same.
	_domain_only_invalidation.assign(mapped_col_count, false);

	return 0;
}

bool Squey::PVScaled::scale_column(PVCol j)
{
	PVScalingFilter::p_type mf = get_properties_for_col(j).get_scaling_filter();
	PVScalingFilter::p_type scaling_filter = mf->clone<PVScalingFilter>();

	const bool on_selection = _selection_domain.has_value() and scale_on_selection(j);

	// A db::array cannot be copied, so the bounds are held either way and only
	// referred to below. The selection is a view onto _selection_domain, which
	// outlives the call.
	pvcop::db::array selection_minmax;
	if (on_selection) {
		selection_minmax = compute_selection_minmax(j);
	}
	const pvcop::db::array& minmax =
	    on_selection ? selection_minmax : get_parent().get_properties_for_col(j).get_minmax();

	// Nothing has moved: the column already sits on these very bounds, under
	// this very regime, and the selection domain is all that asked for it to be
	// looked at again. Recomputing would write the same positions back and pass
	// for a change, which costs a rebuild of every zone tree the column takes
	// part in -- by far the expensive part of a rescaling.
	if (_domain_only_invalidation[j] and on_selection and _scaled_on_domain[j] and
	    _domain_bounds[j].size() == 2 and minmax.size() == 2 and minmax == _domain_bounds[j]) {
		get_properties_for_col(j).set_uptodate();
		return false;
	}

	const pvcop::db::selection domain =
	    on_selection ? pvcop::db::selection(*_selection_domain) : pvcop::db::selection();

	scaling_filter->operator()(
	    get_parent().get_column(j), minmax,
	    get_parent<Squey::PVSource>().get_rushnraw().column(j).invalid_selection(), domain,
	    _scaleds[j].to_core_array<value_type>());

	get_properties_for_col(j).set_uptodate();
	_scaled_on_domain[j] = on_selection ? 1 : 0;
	_domain_bounds[j] = std::move(selection_minmax);

	// The rows holding the ends of the axis, which is what the axis labels
	// read. Under selection scaling the rows outside the domain are pinned to
	// those ends, so several rows share them and the extreme row found over the
	// whole column would name an outside value: look within the domain, whose
	// own extremes are the ends by construction.
	if (on_selection) {
		get_col_minmax(_minmax_values[j].min, _minmax_values[j].max, *_selection_domain, j);
	} else {
		get_col_minmax(_minmax_values[j].min, _minmax_values[j].max, j);
	}

	return true;
}

void Squey::PVScaled::append_scaled()
{
	_columns.emplace_back(PVScalingProperties("default", PVCore::PVArgumentList()));
	_scaleds.emplace_back(Squey::scaling_type, get_row_count());

	// update minmax values
	_minmax_values.resize(_columns.size());
	_scaled_on_domain.resize(_columns.size(), 0);
	_domain_bounds.resize(_columns.size());
	_domain_only_invalidation.resize(_columns.size(), false);
	PVCol col(_columns.size()-1);
	get_col_minmax(_minmax_values[col].min, _minmax_values[col].max, col);
}

void Squey::PVScaled::delete_scaled(PVCol col)
{
	_columns.erase(std::next(_columns.begin(), col.value()));
	_scaleds.erase(std::next(_scaleds.begin(), col.value()));
	_minmax_values.erase(std::next(_minmax_values.begin(), col.value()));
	_scaled_on_domain.erase(std::next(_scaled_on_domain.begin(), col.value()));
	_domain_bounds.erase(std::next(_domain_bounds.begin(), col.value()));
	_domain_only_invalidation.erase(std::next(_domain_only_invalidation.begin(), col.value()));
}

PVRow Squey::PVScaled::get_row_count() const
{
	return get_parent<PVSource>().get_row_count();
}

PVCol Squey::PVScaled::get_nraw_column_count() const
{
	return get_parent<PVMapped>().get_nraw_column_count();
}

QList<PVCol> Squey::PVScaled::get_singleton_columns_indexes()
{
	const PVRow nrows = get_row_count();
	const PVCol ncols = get_nraw_column_count();
	QList<PVCol> cols_ret;

	if (nrows == 0) {
		return cols_ret;
	}

	for (PVCol j(0); j < ncols; j++) {
		const uint32_t* cscaled = get_column_pointer(j);
		const uint32_t ref_v = cscaled[0];
		bool all_same = true;
		for (PVRow i = 1; i < nrows; i++) {
			if (cscaled[i] != ref_v) {
				all_same = false;
				break;
			}
		}
		if (all_same) {
			cols_ret << j;
		}
	}

	return cols_ret;
}

QList<PVCol>
Squey::PVScaled::get_columns_indexes_values_within_range(uint32_t min, uint32_t max, double rate)
{
	const PVRow nrows = get_row_count();
	const PVCol ncols = get_nraw_column_count();
	QList<PVCol> cols_ret;

	if (min > max) {
		return cols_ret;
	}

	auto nrows_d = (double)nrows;
	for (PVCol j(0); j < ncols; j++) {
		PVRow nmatch = 0;
		const uint32_t* cscaled = get_column_pointer(j);
		for (PVRow i = 0; i < nrows; i++) {
			const uint32_t v = cscaled[i];
			if (v >= min && v <= max) {
				nmatch++;
			}
		}
		if ((double)nmatch / nrows_d >= rate) {
			cols_ret << j;
		}
	}

	return cols_ret;
}

QList<PVCol> Squey::PVScaled::get_columns_indexes_values_not_within_range(uint32_t const min,
                                                                            uint32_t const max,
                                                                            double rate)
{
	const PVRow nrows = get_row_count();
	const PVCol ncols = get_nraw_column_count();
	QList<PVCol> cols_ret;

	if (min > max) {
		return cols_ret;
	}

	auto nrows_d = (double)nrows;
	for (PVCol j(0); j < ncols; j++) {
		PVRow nmatch = 0;
		const uint32_t* cscaled = get_column_pointer(j);
		for (PVRow i = 0; i < nrows; i++) {
			const uint32_t v = cscaled[i];
			if (v < min || v > max) {
				nmatch++;
			}
		}
		if ((double)nmatch / nrows_d >= rate) {
			cols_ret << j;
		}
	}

	return cols_ret;
}

void Squey::PVScaled::get_col_minmax(PVRow& min,
                                       PVRow& max,
                                       PVSelection const& sel,
                                       PVCol col) const
{
	const PVRow nrows = get_row_count();

	uint32_t vmin = PVScaled::MAX_VALUE;
	uint32_t vmax = 0;
	min = 0;
	max = 0;

	if (nrows == 0) {
		return;
	}

	// Split into ranges visited in parallel, as the whole-column version is: under
	// selection scaling this runs once per column on every rescaling, and walking
	// the selection on a single thread was costing more than the scaling itself.
	const int thread_count = omp_get_max_threads();
	const PVRow chunk = (nrows + thread_count - 1) / thread_count;

	std::vector<uint32_t> thread_vmin(thread_count, PVScaled::MAX_VALUE);
	std::vector<uint32_t> thread_vmax(thread_count, 0);
	std::vector<PVRow> thread_min(thread_count, 0);
	std::vector<PVRow> thread_max(thread_count, 0);

#pragma omp parallel for
	for (int t = 0; t < thread_count; t++) {
		const PVRow begin = (PVRow)t * chunk;
		const PVRow end = std::min(begin + chunk, nrows);
		if (begin >= end) {
			continue;
		}

		uint32_t local_min = PVScaled::MAX_VALUE;
		uint32_t local_max = 0;
		PVRow local_min_row = 0;
		PVRow local_max_row = 0;

		sel.visit_selected_lines(
		    [&](PVRow i) {
			    const uint32_t v = this->get_value(i, col);
			    if (v > local_max) {
				    local_max = v;
				    local_max_row = i;
			    }
			    if (v < local_min) {
				    local_min = v;
				    local_min_row = i;
			    }
		    },
		    end, begin);

		thread_vmin[t] = local_min;
		thread_vmax[t] = local_max;
		thread_min[t] = local_min_row;
		thread_max[t] = local_max_row;
	}

	for (int t = 0; t < thread_count; t++) {
		if (thread_vmin[t] < vmin) {
			vmin = thread_vmin[t];
			min = thread_min[t];
		}
		if (thread_vmax[t] > vmax) {
			vmax = thread_vmax[t];
			max = thread_max[t];
		}
	}
}

void Squey::PVScaled::get_col_minmax(PVRow& min, PVRow& max, PVCol const col) const
{
	const PVRow nrows = get_row_count();

	min = 0;
	max = 0;

	if (nrows == 0) {
		return;
	}

	// One range per thread, reduced below in range order, as the selection version
	// does. Each thread used to keep a running minimum and maximum and weigh a row
	// against the minimum only when it was no new maximum, which the first row a
	// thread visits always is: a column whose smallest position was held by the
	// first row of a range alone -- row 0 among them -- came back with another row.
	// Reducing in range order rather than in the order the threads finish also
	// settles a tie the same way on every run: the first row holding the value.
	const int thread_count = omp_get_max_threads();
	const PVRow chunk = (nrows + thread_count - 1) / thread_count;
	const uint32_t* const values = get_column_pointer(col);

	std::vector<PVRow> thread_min(thread_count, 0);
	std::vector<PVRow> thread_max(thread_count, 0);

#pragma omp parallel for
	for (int t = 0; t < thread_count; t++) {
		const PVRow begin = (PVRow)t * chunk;
		const PVRow end = std::min(begin + chunk, nrows);
		if (begin >= end) {
			continue;
		}

		// Started from the range's own first row, which is then weighed like any
		// other rather than skipped by a sentinel.
		PVRow local_min = begin;
		PVRow local_max = begin;
		uint32_t vmin = values[begin];
		uint32_t vmax = values[begin];
		for (PVRow i = begin + 1; i < end; i++) {
			const uint32_t v = values[i];
			if (v < vmin) {
				vmin = v;
				local_min = i;
			}
			if (v > vmax) {
				vmax = v;
				local_max = i;
			}
		}

		thread_min[t] = local_min;
		thread_max[t] = local_max;
	}

	for (int t = 0; t < thread_count and (PVRow)t * chunk < nrows; t++) {
		if (values[thread_min[t]] < values[min]) {
			min = thread_min[t];
		}
		if (values[thread_max[t]] > values[max]) {
			max = thread_max[t];
		}
	}
}

PVRow Squey::PVScaled::get_col_min_row(PVCol const c) const
{
	assert(c < get_nraw_column_count());
	return _minmax_values[c].min;
}

PVRow Squey::PVScaled::get_col_max_row(PVCol const c) const
{
	assert(c < get_nraw_column_count());
	return _minmax_values[c].max;
}

pvcop::db::array Squey::PVScaled::compute_selection_minmax(PVCol col) const
{
	assert(_selection_domain.has_value());

	// The bounds of the selection, invalid rows left out of it -- they have a
	// reserved range of their own and would otherwise drag a bound to a value
	// that is not there.
	//
	// Asked of the mapping filter, these bounds would be the ones it declares
	// for the column as a whole: a fixed range for the mappings that have one
	// (a day for 24h, a week for Week), which no selection would ever narrow.
	// Read from the mapped values directly, they are the selection's own.
	const pvcop::db::array& mapped = get_parent().get_column(col);
	const pvcop::db::array& nraw_column = get_parent<Squey::PVSource>().get_rushnraw().column(col);

	// A column with nothing invalid in it has nothing to take out of the domain,
	// and valid_selection() hands back a copy of every row's bit either way.
	if (not nraw_column.invalid_selection()) {
		return pvcop::db::algo::minmax(mapped, *_selection_domain);
	}

	// Held as the owning array it is returned as: a db::selection is a view, and
	// binding one to this temporary would outlive the rows it points at.
	const pvcop::core::memarray<bool> domain = nraw_column.valid_selection(*_selection_domain);

	return pvcop::db::algo::minmax(mapped, domain);
}

bool Squey::PVScaled::scale_on_selection_anywhere() const
{
	if (_scale_on_selection) {
		return true;
	}

	for (PVCol j(0); j < get_nraw_column_count(); j++) {
		if (scale_on_selection(j)) {
			return true;
		}
	}

	return false;
}

bool Squey::PVScaled::refresh_selection_scaling()
{
	_domain_only_invalidation.assign(get_nraw_column_count(), false);

	bool any = false;

	for (PVCol j(0); j < get_nraw_column_count(); j++) {
		const bool on_domain = _selection_domain.has_value() and scale_on_selection(j);

		// Looked at again when it scales over the domain -- the rows it holds may
		// have just changed -- and when it was scaled over the previous one, which
		// is how a column is given the bounds of its every row back.
		if (not on_domain and not _scaled_on_domain[j]) {
			continue;
		}

		// Whether the domain is the only reason this column is not up to date. A
		// column already waiting on something else -- its scaling mode changed, its
		// mapping was recomputed -- has to be recomputed whatever its bounds are.
		_domain_only_invalidation[j] = get_properties_for_col(j).is_uptodate();

		get_properties_for_col(j).invalidate();
		any = true;
	}

	if (not any) {
		return false;
	}

	update_scaling();

	// Columns that turned out to sit where they sat are not in there.
	return not _last_updated_cols.empty();
}

bool Squey::PVScaled::update_scaling_on_selection(PVSelection const& sel)
{
	// Nothing selected leaves nothing to spread over the axis: rather than send
	// every row to the same place, keep the bounds the columns already have.
	if (sel.is_empty()) {
		return false;
	}

	_selection_domain = sel;

	return refresh_selection_scaling();
}

void Squey::PVScaled::clear_selection_domain()
{
	_selection_domain.reset();
	refresh_selection_scaling();
}

void Squey::PVScaled::update_scaling()
{
	create_table();
	_scaled_updated.emit(_last_updated_cols);
}

QList<PVCol> Squey::PVScaled::get_columns_to_update() const
{
	QList<PVCol> ret;

	for (PVCol j(0); j < get_nraw_column_count(); j++) {
		if (!get_properties_for_col(j).is_uptodate()) {
			ret << j;
		}
	}

	return ret;
}

bool Squey::PVScaled::is_uptodate() const
{
	if (!get_parent().is_uptodate()) {
		return false;
	}
	return std::all_of(_columns.begin(), _columns.end(),
	                   std::mem_fn(&PVScalingProperties::is_uptodate));
}

std::string Squey::PVScaled::export_line(PVRow idx,
                                           const PVCore::PVColumnIndexes& col_indexes,
                                           const std::string sep_char,
                                           const std::string) const
{
	assert(col_indexes.size() != 0);

	std::string line;

	for (PVCol c : col_indexes) {
		line += std::to_string(get_value(idx, c)) + sep_char;
	}

	// Remove last sep_char
	line.resize(line.size() - sep_char.size());

	return line;
}

void Squey::PVScaled::serialize_write(PVCore::PVSerializeObject& so) const
{
	so.set_current_status("Saving scaling...");
	QString name = QString::fromStdString(_name);
	so.attribute_write("name", name);

	so.attribute_write("scale_on_selection", _scale_on_selection);
	so.attribute_write("auto_scale_on_selection", _auto_scale_on_selection);

	so.set_current_status("Saving scaling properties...");
	PVCore::PVSerializeObject_p list_prop = so.create_object("properties");

	int idx = 0;
	for (PVScalingProperties const& prop : _columns) {
		PVCore::PVSerializeObject_p new_obj = list_prop->create_object(QString::number(idx++));
		prop.serialize_write(*new_obj);
	}
	so.attribute_write("prop_count", idx);

	// Read the data colletions
	PVCore::PVSerializeObject_p list_obj = so.create_object("view");
	idx = 0;
	for (PVView const* view : get_children()) {
		PVCore::PVSerializeObject_p new_obj = list_obj->create_object(QString::number(idx++));
		view->serialize_write(*new_obj);
	}
	so.attribute_write("view_count", idx);
}

Squey::PVScaled& Squey::PVScaled::serialize_read(PVCore::PVSerializeObject& so,
                                                     Squey::PVMapped& parent)
{
	so.set_current_status("Loading scaling...");
	auto name = so.attribute_read<QString>("name");

	PVCore::PVSerializeObject_p list_prop = so.create_object("properties");

	so.set_current_status("Loading scaling properties...");
	std::list<Squey::PVScalingProperties> columns;
	int prop_count = so.attribute_read<int>("prop_count");
	for (int idx = 0; idx < prop_count; idx++) {
		PVCore::PVSerializeObject_p new_obj = list_prop->create_object(QString::number(idx));
		columns.emplace_back(PVScalingProperties::serialize_read(*new_obj));
	}

	PVScaled& scaled = parent.emplace_add_child(std::move(columns), name.toStdString());

	// Investigations written before selection scaling existed have no such
	// attributes; an absent one reads as false. The domain itself is not saved:
	// the columns are read back scaled over their every row, and a view that
	// finds the setting on scales them over the restored selection.
	scaled._scale_on_selection = so.attribute_read<bool>("scale_on_selection");
	scaled._auto_scale_on_selection = so.attribute_read<bool>("auto_scale_on_selection");

	// Create the list of view
	PVCore::PVSerializeObject_p list_obj = so.create_object("view");

	int view_count = so.attribute_read<int>("view_count");
	for (int idx = 0; idx < view_count; idx++) {
		PVCore::PVSerializeObject_p new_obj = list_obj->create_object(QString::number(idx));
		Squey::PVView::serialize_read(*new_obj, scaled);
	}

	return scaled;
}
