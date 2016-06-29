/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvcop/db/types.h>

#include <QList>
#include <QStringList>
#include <QString>
#include <QVector>

#include <stdlib.h>
#include <limits>

#include <inendi/PVMapped.h>
#include <inendi/PVMapping.h>
#include <inendi/PVPlotting.h>
#include <inendi/PVPlottingFilter.h>
#include <inendi/PVPlotted.h>
#include <inendi/PVSource.h>
#include <inendi/PVSelection.h>
#include <inendi/PVView.h>

#include <tbb/tick_count.h>

#include <iostream>

Inendi::PVPlotted::PVPlotted(PVMapped& mapped)
    : PVCore::PVDataTreeChild<PVMapped, PVPlotted>(mapped), _plotting(this)
{
	create_table();
}

Inendi::PVPlotted::~PVPlotted()
{
	PVLOG_DEBUG("In PVPlotted destructor\n");
}

int Inendi::PVPlotted::create_table()
{
	const PVCol mapped_col_count = get_column_count();
	const PVRow nrows = get_row_count();

	// Transposed normalized unisnged integer.
	// Align the number of lines on a mulitple of 4, in order to have 16-byte
	// aligned starting adresses for each axis

	const PVRow nrows_aligned = get_aligned_row_count();
	_uint_table.resize((size_t)mapped_col_count * (size_t)nrows_aligned);

	_last_updated_cols.clear();
	_minmax_values.resize(mapped_col_count);

	for (PVCol j = 0; j < mapped_col_count; j++) {
		if (_plotting.is_col_uptodate(j)) {
			continue;
		}
		PVPlottingFilter::p_type mf = _plotting.get_filter_for_col(j);
		PVPlottingFilter::p_type plotting_filter = mf->clone<PVPlottingFilter>();

		plotting_filter->set_dest_array(nrows, get_column_pointer(j));

		boost::this_thread::interruption_point();

		plotting_filter->operator()(
		    get_parent().get_column(j),
		    get_parent().get_mapping().get_properties_for_col(j).get_minmax());

		boost::this_thread::interruption_point();
		_plotting.set_uptodate_for_col(j);
		_last_updated_cols.push_back(j);

		get_col_minmax(_minmax_values[j].min, _minmax_values[j].max, j);
	}

	return 0;
}

PVRow Inendi::PVPlotted::get_row_count() const
{
	return get_parent<PVSource>().get_row_count();
}

PVCol Inendi::PVPlotted::get_column_count() const
{
	return get_parent<PVMapped>().get_column_count();
}

QList<PVCol> Inendi::PVPlotted::get_singleton_columns_indexes()
{
	const PVRow nrows = get_row_count();
	const PVCol ncols = get_column_count();
	QList<PVCol> cols_ret;

	if (nrows == 0) {
		return cols_ret;
	}

	for (PVCol j = 0; j < ncols; j++) {
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
	const PVCol ncols = get_column_count();
	QList<PVCol> cols_ret;

	if (min > max) {
		return cols_ret;
	}

	double nrows_d = (double)nrows;
	for (PVCol j = 0; j < ncols; j++) {
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
	const PVCol ncols = get_column_count();
	QList<PVCol> cols_ret;

	if (min > max) {
		return cols_ret;
	}

	double nrows_d = (double)nrows;
	for (PVCol j = 0; j < ncols; j++) {
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

	// We need to swap as the plotted has been reversed
	std::swap(local_min, local_max);

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
		uint32_t local_min = 0;
		uint32_t local_max = PVPlotted::MAX_VALUE;
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
			} else if (local_max > vmax) {
				vmax = local_max;
				max = local_max_col;
			}
		}
	}

	// We need to swap as the plotted has been reversed
	std::swap(min, max);
}

PVRow Inendi::PVPlotted::get_col_min_row(PVCol const c) const
{
	assert(c < get_column_count());
	return _minmax_values[c].min;
}

PVRow Inendi::PVPlotted::get_col_max_row(PVCol const c) const
{
	assert(c < get_column_count());
	return _minmax_values[c].max;
}

void Inendi::PVPlotted::update_plotting()
{
	create_table();
	_plotted_updated.emit();
}

bool Inendi::PVPlotted::is_uptodate() const
{
	if (!get_parent().is_uptodate()) {
		return false;
	}

	return _plotting.is_uptodate();
}

bool Inendi::PVPlotted::is_current_plotted() const
{
	Inendi::PVView const* cur_view = get_parent<PVSource>().current_view();
	auto children = get_children();
	return std::find(children.begin(), children.end(), cur_view) != children.end();
}

void Inendi::PVPlotted::finish_process_from_rush_pipeline()
{
	for (auto view : get_children()) {
		view->finish_process_from_rush_pipeline();
	}
}

QList<PVCol> Inendi::PVPlotted::get_columns_to_update() const
{
	QList<PVCol> ret;

	for (PVCol j = 0; j < get_column_count(); j++) {
		if (!_plotting.is_col_uptodate(j)) {
			ret << j;
		}
	}

	return ret;
}

void Inendi::PVPlotted::serialize_write(PVCore::PVSerializeObject& so)
{
	so.object("plotting", _plotting, QString(), false, nullptr, false);

	// Read the data colletions
	PVCore::PVSerializeObject_p list_obj =
	    so.create_object(get_children_serialize_name(), get_children_description(), true, true);
	int idx = 0;
	for (PVView* view : get_children()) {
		QString child_name = QString::number(idx++);
		PVCore::PVSerializeObject_p new_obj = list_obj->create_object(
		    child_name, QString::fromStdString(view->get_serialize_description()), false);
		view->serialize(*new_obj, so.get_version());
		new_obj->_bound_obj = view;
		new_obj->_bound_obj_type = typeid(PVView);
	}
}

void Inendi::PVPlotted::serialize_read(PVCore::PVSerializeObject& so)
{
	// Create the list of view
	PVCore::PVSerializeObject_p list_obj =
	    so.create_object(get_children_serialize_name(), get_children_description(), true, true);
	int idx = 0;
	try {
		while (true) {
			// FIXME It throws when there are no more data collections.
			// It should not be an exception as it is a normal behavior.
			PVCore::PVSerializeObject_p new_obj = list_obj->create_object(QString::number(idx));
			PVView& view = emplace_add_child();
			view.serialize(*new_obj, so.get_version());
			new_obj->_bound_obj = &view;
			new_obj->_bound_obj_type = typeid(PVView);
			idx++;
		}
	} catch (PVCore::PVSerializeArchiveErrorNoObject const&) {
		return;
	}
}
