/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <inendi/PVMapped.h>             // for PVMapped
#include <inendi/PVMappingProperties.h>  // for PVMappingProperties
#include <inendi/PVPlotted.h>            // for PVPlotted, etc
#include <inendi/PVPlottingFilter.h>     // for PVPlottingFilter, etc
#include <inendi/PVPlottingProperties.h> // for PVPlottingProperties
#include <inendi/PVSelection.h>          // for PVSelection
#include <inendi/PVSource.h>             // for PVSource
#include <inendi/PVView.h>               // for PVView

#include <pvkernel/rush/PVFormat.h> // for PVFormat

#include <pvkernel/core/PVColumnIndexes.h>   // for PVColumnIndexes
#include <pvkernel/core/PVDataTreeObject.h>  // for PVDataTreeChild
#include <pvkernel/core/PVLogger.h>          // for PVLOG_DEBUG
#include <pvkernel/core/PVSerializeObject.h> // for PVSerializeObject_p, etc

#include <pvbase/types.h> // for PVCol, PVRow

#include <boost/thread/thread.hpp>

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

Inendi::PVPlotted::PVPlotted(PVMapped& mapped, std::string const& name)
    : PVCore::PVDataTreeChild<PVMapped, PVPlotted>(mapped), _name(name)
{
	PVRush::PVFormat const& format = get_parent<Inendi::PVSource>().get_format();

	for (PVCol i(0); i < format.get_axes().size(); i++) {
		_columns.emplace_back(format, i);
	}
	create_table();
}

Inendi::PVPlotted::PVPlotted(PVMapped& mapped,
                             std::list<Inendi::PVPlottingProperties>&& column,
                             std::string const& name)
    : PVCore::PVDataTreeChild<PVMapped, PVPlotted>(mapped), _columns(std::move(column)), _name(name)
{
	create_table();
}

Inendi::PVPlotted::~PVPlotted()
{
	PVLOG_DEBUG("In PVPlotted destructor\n");
}

int Inendi::PVPlotted::create_table()
{
	const PVCol mapped_col_count = get_nraw_column_count();

	for (size_t i = 0; i < _columns.size(); i++) {
		_plotteds.emplace_back(Inendi::plotting_type, get_row_count());
	}

	_last_updated_cols.clear();
	_minmax_values.resize(mapped_col_count);

	auto const& axes_format = get_parent<Inendi::PVSource>().get_format().get_axes();

	for (PVCol j(0); j < mapped_col_count; j++) {
		auto const& mapping_mode = get_parent<PVMapped>().get_properties_for_col(j).get_mode();
		auto usable_pt = get_properties_for_col(j).get_plotting_filter()->list_usable_type();
		if (not usable_pt.empty() and
		    usable_pt.count(
		        std::make_pair(axes_format[j].get_type().toStdString(), mapping_mode)) == 0) {
			get_properties_for_col(j).set_mode("default");
		}

		if (get_properties_for_col(j).is_uptodate()) {
			continue;
		}
		PVPlottingFilter::p_type mf = get_properties_for_col(j).get_plotting_filter();
		PVPlottingFilter::p_type plotting_filter = mf->clone<PVPlottingFilter>();

		boost::this_thread::interruption_point();

		plotting_filter->operator()(
		    get_parent().get_column(j), get_parent().get_properties_for_col(j).get_minmax(),
		    get_parent<Inendi::PVSource>().get_rushnraw().column(j).invalid_selection(),
		    _plotteds[j].to_core_array<value_type>());

		boost::this_thread::interruption_point();
		get_properties_for_col(j).set_uptodate();
		_last_updated_cols.push_back(j);

		get_col_minmax(_minmax_values[j].min, _minmax_values[j].max, j);
	}

	return 0;
}

void Inendi::PVPlotted::append_plotted()
{
	_columns.emplace_back(PVPlottingProperties("default", PVCore::PVArgumentList()));
	_plotteds.emplace_back(Inendi::plotting_type, get_row_count());

	// update minmax values
	_minmax_values.resize(_columns.size());
	PVCol col(_columns.size()-1);
	get_col_minmax(_minmax_values[col].min, _minmax_values[col].max, col);
}

void Inendi::PVPlotted::delete_plotted(PVCol col)
{
	_columns.erase(std::next(_columns.begin(), col.value()));
	_plotteds.erase(std::next(_plotteds.begin(), col.value()));
	_minmax_values.erase(std::next(_minmax_values.begin(), col.value()));
}

PVRow Inendi::PVPlotted::get_row_count() const
{
	return get_parent<PVSource>().get_row_count();
}

PVCol Inendi::PVPlotted::get_nraw_column_count() const
{
	return get_parent<PVMapped>().get_nraw_column_count();
}

QList<PVCol> Inendi::PVPlotted::get_singleton_columns_indexes()
{
	const PVRow nrows = get_row_count();
	const PVCol ncols = get_nraw_column_count();
	QList<PVCol> cols_ret;

	if (nrows == 0) {
		return cols_ret;
	}

	for (PVCol j(0); j < ncols; j++) {
		const uint32_t* cplotted = get_column_pointer(j);
		const uint32_t ref_v = cplotted[0];
		bool all_same = true;
		for (PVRow i = 1; i < nrows; i++) {
			if (cplotted[i] != ref_v) {
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
Inendi::PVPlotted::get_columns_indexes_values_within_range(uint32_t min, uint32_t max, double rate)
{
	const PVRow nrows = get_row_count();
	const PVCol ncols = get_nraw_column_count();
	QList<PVCol> cols_ret;

	if (min > max) {
		return cols_ret;
	}

	double nrows_d = (double)nrows;
	for (PVCol j(0); j < ncols; j++) {
		PVRow nmatch = 0;
		const uint32_t* cplotted = get_column_pointer(j);
		for (PVRow i = 0; i < nrows; i++) {
			const uint32_t v = cplotted[i];
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

QList<PVCol> Inendi::PVPlotted::get_columns_indexes_values_not_within_range(uint32_t const min,
                                                                            uint32_t const max,
                                                                            double rate)
{
	const PVRow nrows = get_row_count();
	const PVCol ncols = get_nraw_column_count();
	QList<PVCol> cols_ret;

	if (min > max) {
		return cols_ret;
	}

	double nrows_d = (double)nrows;
	for (PVCol j(0); j < ncols; j++) {
		PVRow nmatch = 0;
		const uint32_t* cplotted = get_column_pointer(j);
		for (PVRow i = 0; i < nrows; i++) {
			const uint32_t v = cplotted[i];
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

void Inendi::PVPlotted::get_col_minmax(PVRow& min,
                                       PVRow& max,
                                       PVSelection const& sel,
                                       PVCol col) const
{
	PVRow local_min, local_max;
	uint32_t vmin, vmax;
	vmin = PVPlotted::MAX_VALUE;
	vmax = 0;
	local_min = 0;
	local_max = 0;
	sel.visit_selected_lines(
	    [&](PVRow i) {
		    const uint32_t v = this->get_value(i, col);
		    if (v > vmax) {
			    vmax = v;
			    local_max = i;
		    }
		    if (v < vmin) {
			    vmin = v;
			    local_min = i;
		    }
		},
	    get_row_count());

	min = local_min;
	max = local_max;
}

/******************************************************************************
 * get_col_minmax
 *
 * Use a parallele reduction to compute the indices of the row with min and max
 * value
 *****************************************************************************/

void Inendi::PVPlotted::get_col_minmax(PVRow& min, PVRow& max, PVCol const col) const
{
	uint32_t vmin = PVPlotted::MAX_VALUE;
	uint32_t vmax = 0;
	const PVRow nrows = get_row_count();
// TODO: use the SSE4.2 optimised version here
#pragma omp parallel
	{
		// Define thread local variables for local minmax extraction
		uint32_t local_min = PVPlotted::MAX_VALUE;
		uint32_t local_max = 0;
		PVRow local_min_col = 0;
		PVRow local_max_col = 0;

// Share work among threads
#pragma omp for
		for (PVRow i = 0; i < nrows; i++) {
			const uint32_t v = this->get_value(i, col);
			if (v > local_max) {
				local_max = v;
				local_max_col = i;
			} else if (v < local_min) {
				local_min = v;
				local_min_col = i;
			}
		}

// Perform final reduction. This is not a parallel reduction but it should
// not be really expensive.
// TODO : As it is a two arguments reduction, it can be done using
// OpenMP 3.1 but maybe with custom reduction from OpenMP 4.0
#pragma omp critical
		{
			if (local_min < vmin) {
				vmin = local_min;
				min = local_min_col;
			}
			if (local_max > vmax) {
				vmax = local_max;
				max = local_max_col;
			}
		}
	}
}

PVRow Inendi::PVPlotted::get_col_min_row(PVCol const c) const
{
	assert(c < get_nraw_column_count());
	return _minmax_values[c].min;
}

PVRow Inendi::PVPlotted::get_col_max_row(PVCol const c) const
{
	assert(c < get_nraw_column_count());
	return _minmax_values[c].max;
}

void Inendi::PVPlotted::update_plotting()
{
	create_table();
	_plotted_updated.emit(_last_updated_cols);
}

QList<PVCol> Inendi::PVPlotted::get_columns_to_update() const
{
	QList<PVCol> ret;

	for (PVCol j(0); j < get_nraw_column_count(); j++) {
		if (!get_properties_for_col(j).is_uptodate()) {
			ret << j;
		}
	}

	return ret;
}

bool Inendi::PVPlotted::is_uptodate() const
{
	if (!get_parent().is_uptodate()) {
		return false;
	}
	return std::all_of(_columns.begin(), _columns.end(),
	                   std::mem_fn(&PVPlottingProperties::is_uptodate));
}

std::string Inendi::PVPlotted::export_line(PVRow idx,
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

void Inendi::PVPlotted::serialize_write(PVCore::PVSerializeObject& so) const
{
	so.set_current_status("Saving plotting...");
	QString name = QString::fromStdString(_name);
	so.attribute_write("name", name);

	so.set_current_status("Saving plotting properties...");
	PVCore::PVSerializeObject_p list_prop = so.create_object("properties");

	int idx = 0;
	for (PVPlottingProperties const& prop : _columns) {
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

Inendi::PVPlotted& Inendi::PVPlotted::serialize_read(PVCore::PVSerializeObject& so,
                                                     Inendi::PVMapped& parent)
{
	so.set_current_status("Loading plotting...");
	QString name = so.attribute_read<QString>("name");

	PVCore::PVSerializeObject_p list_prop = so.create_object("properties");

	so.set_current_status("Loading plotting properties...");
	std::list<Inendi::PVPlottingProperties> columns;
	int prop_count = so.attribute_read<int>("prop_count");
	for (int idx = 0; idx < prop_count; idx++) {
		PVCore::PVSerializeObject_p new_obj = list_prop->create_object(QString::number(idx));
		columns.emplace_back(PVPlottingProperties::serialize_read(*new_obj));
	}

	PVPlotted& plotted = parent.emplace_add_child(std::move(columns), name.toStdString());

	// Create the list of view
	PVCore::PVSerializeObject_p list_obj = so.create_object("view");

	int view_count = so.attribute_read<int>("view_count");
	for (int idx = 0; idx < view_count; idx++) {
		PVCore::PVSerializeObject_p new_obj = list_obj->create_object(QString::number(idx));
		Inendi::PVView::serialize_read(*new_obj, plotted);
	}

	return plotted;
}
