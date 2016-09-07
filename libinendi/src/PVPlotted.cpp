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
#include <inendi/PVPlottingFilter.h>
#include <inendi/PVPlotted.h>
#include <inendi/PVSource.h>
#include <inendi/PVSelection.h>
#include <inendi/PVView.h>

#include <tbb/tick_count.h>

#include <iostream>

Inendi::PVPlotted::PVPlotted(PVMapped& mapped, std::string const& name)
    : PVCore::PVDataTreeChild<PVMapped, PVPlotted>(mapped), _name(name)
{
	PVRush::PVFormat const& format = get_parent<Inendi::PVSource>().get_format();

	for (int i = 0; i < format.get_axes().size(); i++) {
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

	// Transposed normalized unisnged integer.
	// Align the number of lines on a mulitple of 4, in order to have 16-byte
	// aligned starting adresses for each axis

	const PVRow nrows_aligned = get_aligned_row_count();
	_uint_table.resize((size_t)mapped_col_count * (size_t)nrows_aligned);

	_last_updated_cols.clear();
	_minmax_values.resize(mapped_col_count);

	for (PVCol j = 0; j < mapped_col_count; j++) {
		if (get_properties_for_col(j).is_uptodate()) {
			continue;
		}
		PVPlottingFilter::p_type mf = get_properties_for_col(j).get_plotting_filter();
		PVPlottingFilter::p_type plotting_filter = mf->clone<PVPlottingFilter>();

		boost::this_thread::interruption_point();

		plotting_filter->operator()(get_parent().get_column(j),
		                            get_parent().get_properties_for_col(j).get_minmax(),
		                            get_column_pointer(j));

		boost::this_thread::interruption_point();
		get_properties_for_col(j).set_uptodate();
		_last_updated_cols.push_back(j);

		get_col_minmax(_minmax_values[j].min, _minmax_values[j].max, j);
	}

	return 0;
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
	const PVCol ncols = get_nraw_column_count();
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
	const PVCol ncols = get_nraw_column_count();
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
	_plotted_updated.emit();
}

QList<PVCol> Inendi::PVPlotted::get_columns_to_update() const
{
	QList<PVCol> ret;

	for (PVCol j = 0; j < get_nraw_column_count(); j++) {
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

	for (int c : col_indexes) {
		line += std::to_string(get_value(idx, c)) + sep_char;
	}

	// Remove last sep_char
	line.resize(line.size() - sep_char.size());

	return line;
}

void Inendi::PVPlotted::serialize_write(PVCore::PVSerializeObject& so)
{
	so.set_current_status("Serialize Plotting.");
	QString name = QString::fromStdString(_name);
	so.attribute("name", name);

	so.set_current_status("Serialize Plotting properties.");
	PVCore::PVSerializeObject_p list_prop =
	    so.create_object("properties", "plotting properties", true, true);

	int idx = 0;
	for (PVPlottingProperties& prop : _columns) {
		QString child_name = QString::number(idx++);
		PVCore::PVSerializeObject_p new_obj = list_prop->create_object(child_name, "", false);
		prop.serialize_write(*new_obj);
		new_obj->set_bound_obj(prop);
	}
	so.attribute("prop_count", idx);

	// Read the data colletions
	PVCore::PVSerializeObject_p list_obj =
	    so.create_object(get_children_serialize_name(), get_children_description(), true, true);
	idx = 0;
	for (PVView* view : get_children()) {
		QString child_name = QString::number(idx++);
		PVCore::PVSerializeObject_p new_obj = list_obj->create_object(
		    child_name, QString::fromStdString(view->get_serialize_description()), false);
		view->serialize_write(*new_obj);
		new_obj->set_bound_obj(*view);
	}
	so.attribute("view_count", idx);
}

Inendi::PVPlotted& Inendi::PVPlotted::serialize_read(PVCore::PVSerializeObject& so,
                                                     Inendi::PVMapped& parent)
{
	so.set_current_status("Load plotting");
	QString name;
	so.attribute("name", name);

	PVCore::PVSerializeObject_p list_prop = so.create_object("properties", "", true, true);

	so.set_current_status("Load plotting properties");
	std::list<Inendi::PVPlottingProperties> columns;
	int prop_count;
	so.attribute("prop_count", prop_count);
	for (int idx = 0; idx < prop_count; idx++) {
		PVCore::PVSerializeObject_p new_obj = list_prop->create_object(QString::number(idx));
		columns.emplace_back(PVPlottingProperties::serialize_read(*new_obj));
	}

	PVPlotted& plotted = parent.emplace_add_child(std::move(columns), name.toStdString());

	// Create the list of view
	PVCore::PVSerializeObject_p list_obj = so.create_object(
	    plotted.get_children_serialize_name(), plotted.get_children_description(), true, true);

	int view_count;
	so.attribute("view_count", view_count);
	for (int idx = 0; idx < view_count; idx++) {
		PVCore::PVSerializeObject_p new_obj = list_obj->create_object(QString::number(idx));
		Inendi::PVView::serialize_read(*new_obj, plotted);
	}

	return plotted;
}
