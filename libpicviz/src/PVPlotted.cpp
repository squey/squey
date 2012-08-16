/**
 * \file PVPlotted.cpp
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#include <QList>
#include <QStringList>
#include <QString>
#include <QVector>

#include <stdlib.h>
#include <float.h>

#include <picviz/PVMapped.h>
#include <picviz/PVMapping.h>
#include <picviz/PVPlotting.h>
#include <picviz/PVPlottingFilter.h>
#include <picviz/PVPlotted.h>
#include <picviz/PVSource.h>
#include <picviz/PVSelection.h>
#include <picviz/PVView.h>

#include <tbb/tick_count.h>

#include <iostream>

#include <omp.h>

#define dbg { std::cout<<"*** CUDA TRACE ***  file "<<__FILE__<<"  line "<<__LINE__<<std::endl; }

Picviz::PVPlotted::PVPlotted()
{
	//process_from_parent_mapped(false);
}

Picviz::PVPlotted::~PVPlotted()
{
	PVLOG_INFO("In PVPlotted destructor\n");
}

void Picviz::PVPlotted::set_parent_from_ptr(PVMapped* mapped)
{
	data_tree_plotted_t::set_parent_from_ptr(mapped);

	if (!_plotting) {
		_plotting.reset(new PVPlotting(this));
	}

	// Set parent mapping for properties
	QList<PVPlottingProperties>::iterator it;
	for (it = _plotting->_columns.begin(); it != _plotting->_columns.end(); it++) {
		it->set_mapping(*get_parent()->get_mapping());
	}
}

int Picviz::PVPlotted::create_table()
{
	const PVCol mapped_col_count = get_column_count();
	const PVRow nrows = get_row_count();

	// That buffer will be about 3.8MB for 10 million lines, so we keep it
	// to save some allocations.
	_tmp_values.reserve(nrows);
	float* p_tmp_v = &_tmp_values[0];
	
	_table.resize(mapped_col_count * nrows);

	// Futur and only plotted, transposed normalized unisnged integer.
	// Align the number of lines on a mulitple of 4, in order to have 16-byte aligned starting adresses for each axis
	
	const PVRow nrows_aligned = get_aligned_row_count();
	_uint_table.resize(mapped_col_count * nrows_aligned);
	
	tbb::tick_count tstart = tbb::tick_count::now();
	
	// We will use the trans_table of PVMapped and write "in live" in
	// the "table" of PVPlotted
	
	// Create our own plugins from the library
	std::vector<PVPlottingFilter::p_type> plotting_filters;
	plotting_filters.resize(mapped_col_count);
	for (PVCol j = 0; j < mapped_col_count; j++) {
		PVPlottingFilter::p_type mf = _plotting->get_filter_for_col(j);
		if (mf) {
			plotting_filters[j] = mf->clone<PVPlottingFilter>();
		}
	}

	try {

		PVLOG_INFO("(PVPlotted::create_table) begin parallel plotting\n");
		for (PVCol j = 0; j < mapped_col_count; j++) {
			if (_plotting->is_col_uptodate(j)) {
				continue;
			}
			PVPlottingFilter::p_type plotting_filter = plotting_filters[j];
			if (!plotting_filter) {
				PVLOG_ERROR("No valid plotting filter function is defined for axis %d !\n", j);
				continue;
			}

			plotting_filter->set_mapping_mode(get_parent()->get_mapping()->get_mode_for_col(j));
			plotting_filter->set_mandatory_params(get_parent()->get_mapping()->get_mandatory_params_for_col(j));
			plotting_filter->set_dest_array(nrows, p_tmp_v);
			boost::this_thread::interruption_point();
			tbb::tick_count plstart = tbb::tick_count::now();
			plotting_filter->operator()(get_parent()->trans_table.getRowData(j));
			tbb::tick_count plend = tbb::tick_count::now();
			int64_t nrows_tmp = nrows;

			PVLOG_INFO("(PVPlotted::create_table) parallel plotting for axis %d took %0.4f seconds, plugin was %s.\n", j, (plend-plstart).seconds(), qPrintable(plotting_filter->registered_name()));

			boost::this_thread::interruption_point();
//#pragma omp parallel for
			for (int64_t i = 0; i < nrows_tmp; i++) {
				float v = p_tmp_v[i];
				_table[i*mapped_col_count+j] = v;
				_uint_table[j*nrows_aligned + i] = (uint32_t) ((double)v * (double)UINT_MAX);

#ifndef NDEBUG
				// Check that every plotted value is between 0 and 1
				if (v > 1 || v < 0) {
					PVLOG_WARN("Plotting value for row/col %d/%d is %0.4f !\n", i,j,v);
				}
#endif
			}

			_plotting->set_uptodate_for_col(j);
		}
		PVLOG_INFO("(PVPlotted::create_table) end parallel plotting\n");

		tbb::tick_count tend = tbb::tick_count::now();
		PVLOG_INFO("(PVPlotted::create_table) plotting took %0.4f seconds.\n", (tend-tstart).seconds());
	}
	catch (boost::thread_interrupted const& e)
	{
		PVLOG_INFO("(PVPlotted::create_table) plotting canceled.\n");
		throw e;
	}

	return 0;
}

void Picviz::PVPlotted::process_expanded_selections()
{
	list_expanded_selection_t::const_iterator it;
	for (it = _expanded_sels.begin(); it != _expanded_sels.end(); it++) {
		expand_selection_on_axis(*(it->sel_p), it->col, it->type, false);
	}
}

void Picviz::PVPlotted::expand_selection_on_axis(PVSelection const& sel, PVCol axis_id, QString const& mode, bool add)
{
	// Recompute a part of the plotted by expanding a selection through the whole axis
	//
	
	// Get axis type
	QString plugin = _plotting->get_properties_for_col(axis_id).get_type() + "_" + mode;
	PVPlottingFilter::p_type filter = LIB_CLASS(PVPlottingFilter)::get().get_class_by_name(plugin);
	if (!filter) {
		return;
	}
	PVPlottingFilter::p_type filter_clone = filter->clone<PVPlottingFilter>();

	PVCol ncols = get_column_count();
	assert(axis_id < ncols);
	PVMapped::mapped_sub_col_t sub_plotted;
	float min,max;
	get_parent<PVMapped>()->get_sub_col_minmax(sub_plotted, min, max, sel, axis_id);
	if (sub_plotted.size() == 0 || min >= max) {
		return;
	}
	PVMapped::mapped_sub_col_t::const_iterator it;
	filter_clone->init_expand(min, max);
	for (it = sub_plotted.begin(); it != sub_plotted.end(); it++) {
		_table[it->first*ncols+axis_id] = filter_clone->expand_plotted(it->second);
	}
	if (add) {
		_expanded_sels.push_back(ExpandedSelection(axis_id, sel, mode));
	}
}

bool Picviz::PVPlotted::dump_buffer_to_file(QString const& file, bool write_as_transposed) const
{
	// Dump the plotted buffer into a file
	// Format is:
	//  * 4 bytes: number of columns
	//  * 1 byte: is it written in a transposed form
	//  * the rest is the plotted
	
	QFile f(file);
	if (!f.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
		PVLOG_ERROR("Error while opening %s for writing: %s.\n", qPrintable(file), qPrintable(f.errorString()));
		return false;
	}

	PVCol ncols = get_column_count();
	f.write((const char*) &ncols, sizeof(ncols));
	f.write((const char*) &write_as_transposed, sizeof(bool));

	const float* buf_to_write = &_table[0];
	PVCore::PVMatrix<float, PVCol, PVRow> transp_plotted;
	if (write_as_transposed) {
		PVCore::PVMatrix<float, PVRow, PVCol> matrix_plotted;
		matrix_plotted.set_raw_buffer((float*) buf_to_write, get_row_count(), get_column_count());
		matrix_plotted.transpose_to(transp_plotted);
		buf_to_write = transp_plotted.get_row_ptr(0);
	}
	
	ssize_t sbuf = _table.size()*sizeof(float);
	if (f.write((const char*) buf_to_write, sbuf) != sbuf) {
		PVLOG_ERROR("Error while writing '%s': %s.\n", qPrintable(file), qPrintable(f.errorString()));
		return false;
	}
	f.close();

	return true;
}

bool Picviz::PVPlotted::load_buffer_from_file(plotted_table_t& buf, PVCol& ncols, bool get_transposed_version, QString const& file)
{
	ncols = 0;

	FILE* f = fopen(qPrintable(file), "r");
	if (!f) {
		PVLOG_ERROR("Error while opening %s for writing: %s.\n", qPrintable(file), strerror(errno));
		return false;
	}

	static_assert(sizeof(off_t) == sizeof(uint64_t), "sizeof(off_t) != sizeof(uint64_t). Please define -D_FILE_OFFSET_BITS=64");

	// Get file size
	fseek(f, 0, SEEK_END);
	const uint64_t fsize = ftello(f);
	fseek(f, 0, SEEK_SET);

	ssize_t size_buf = fsize-sizeof(PVCol)-sizeof(bool);
	if (size_buf <= 0) {
		fclose(f);
		PVLOG_ERROR("File is too small to be valid !\n");
		return false;
	}

	if (fread((void*) &ncols, sizeof(PVCol), 1, f) != 1) {
		PVLOG_ERROR("File is too small to be valid !\n");
		fclose(f);
		return false;
	}
	bool is_transposed = false;
	if (fread((char*) &is_transposed, sizeof(bool), 1, f) != 1) {
		PVLOG_ERROR("Error while reading '%s': %s.\n", qPrintable(file), strerror(errno));
		fclose(f);
		return false;
	}

	bool must_transpose = (is_transposed != get_transposed_version);

	size_t nfloats = size_buf/sizeof(float);
	size_t size_read = nfloats*sizeof(float);
	buf.resize(nfloats);

	PVLOG_INFO("(Picviz::load_buffer_from_file) number of cols: %d , nfloats: %u, nrows: %u\n", ncols, nfloats, nfloats/ncols);

	float* dest_buf = &buf[0];
	if (must_transpose) {
		dest_buf = (float*) malloc(size_read);
	}

	if (fread((void*) dest_buf, sizeof(float), nfloats, f) != nfloats) {
		PVLOG_ERROR("Error while reading '%s': %s.\n", qPrintable(file), strerror(errno));
		return false;
	}

	if (must_transpose) {
		if (is_transposed) {
			PVCore::PVMatrix<float, PVRow, PVCol> final;
			PVCore::PVMatrix<float, PVCol, PVRow> org;
			org.set_raw_buffer(dest_buf, ncols, nfloats/ncols);
			
			org.transpose_to(final);
		}
		else {
			PVCore::PVMatrix<float, PVRow, PVCol> org;
			PVCore::PVMatrix<float, PVCol, PVRow> final;
			org.set_raw_buffer(dest_buf, nfloats/ncols, ncols);
			final.set_raw_buffer(&buf[0], ncols, nfloats/ncols);
			org.transpose_to(final);
		}
		free(dest_buf);
	}

	fclose(f);

	return true;
}

PVRow Picviz::PVPlotted::get_row_count() const
{
	return get_parent<PVMapped>()->get_row_count();
}

PVCol Picviz::PVPlotted::get_column_count() const
{
	return get_parent<PVMapped>()->get_column_count();
}

PVRush::PVNraw::nraw_table& Picviz::PVPlotted::get_qtnraw()
{
	return get_parent<PVSource>()->get_qtnraw();
}

const PVRush::PVNraw::nraw_table& Picviz::PVPlotted::get_qtnraw() const
{
	return get_parent<PVSource>()->get_qtnraw();
}

float Picviz::PVPlotted::get_value(PVRow row, PVCol col) const
{
	return _table[row * get_column_count() + col];
}

void Picviz::PVPlotted::to_csv()
{
	PVRow row_count;
	PVCol col_count;

	row_count = get_row_count();
	col_count = get_column_count();

	for (PVRow r = 0; r < row_count; r++) {
		for (PVCol c = 0; c < col_count; c++) {
			printf("%.6f", get_value(r,c));
			if (c!=col_count-1) {
				std::cout << "|";
			}
		}
		std::cout << "\n";
	}

}

QList<PVCol> Picviz::PVPlotted::get_singleton_columns_indexes()
{
	const PVRow nrows = get_row_count();
	const PVCol ncols = get_column_count();
	QList<PVCol> cols_ret;

	if (nrows == 0) {
		return cols_ret;
	}

	for (PVCol j = 0; j < ncols; j++) {
		//const float* values = trans_table.getRowData(j);
		//const float ref_v = values[0];
		// Well, that's completely cache non-optimised, as we're reading the table in the wrong side.
		// But, with 2GB a table of 50 columnsx10 million lines, can we afford to keep the transposed version ?
		const float ref_v = _table[j];
		bool all_same = true;
		for (PVRow i = 1; i < nrows; i++) {
			if (_table[ncols*i+j] != ref_v) {
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

QList<PVCol> Picviz::PVPlotted::get_columns_indexes_values_within_range(float min, float max, double rate)
{
	const PVRow nrows = get_row_count();
	const PVCol ncols = get_column_count();
	QList<PVCol> cols_ret;

	if (min > max) {
		return cols_ret;
	}

	double nrows_d = (double) nrows;
	for (PVCol j = 0; j < ncols; j++) {
		PVRow nmatch = 0;
		//const float* values = trans_table.getRowData(j);
		for (PVRow i = 0; i < nrows; i++) {
			//const float v = values[i];
			const float v = _table[ncols*i+j];
			if (v >= min && v <= max) {
				nmatch++;
			}
		}
		if ((double)nmatch/nrows_d >= rate) {
			cols_ret << j;
		}
	}

	return cols_ret;
}

QList<PVCol> Picviz::PVPlotted::get_columns_indexes_values_not_within_range(float min, float max, double rate)
{
	const PVRow nrows = get_row_count();
	const PVCol ncols = get_column_count();
	QList<PVCol> cols_ret;

	if (min > max) {
		return cols_ret;
	}

	double nrows_d = (double) nrows;
	for (PVCol j = 0; j < ncols; j++) {
		PVRow nmatch = 0;
		//const float* values = trans_table.getRowData(j);
		for (PVRow i = 0; i < nrows; i++) {
			//const float v = values[i];
			const float v = _table[ncols*i+j];
			if (v < min || v > max) {
				nmatch++;
			}
		}
		if ((double)nmatch/nrows_d >= rate) {
			cols_ret << j;
		}
	}

	return cols_ret;
}

void Picviz::PVPlotted::get_sub_col_minmax(plotted_sub_col_t& ret, float& min, float& max, PVSelection const& sel, PVCol col) const
{
	min = FLT_MAX;
	max = 0;
	PVRow size = get_qtnraw().get_nrows();
	ret.reserve(sel.get_number_of_selected_lines_in_range(0, size-1));
	for (PVRow i = 0; i < size; i++) {
		if (sel.get_line(i)) {
			float v = get_value(i, col);
			if (v > max) {
			   max = v;
			}
	 		if (v < min) {
				min = v;
			}		
			ret.push_back(plotted_sub_col_t::value_type(i, v));
		}
	}
}

void Picviz::PVPlotted::get_col_minmax(PVRow& min, PVRow& max, PVSelection const& sel, PVCol col) const
{
	float vmin,vmax;
	vmin = FLT_MAX;
	vmax = 0;
	min = 0;
	max = 0;
	for (PVRow i = 0; i < get_qtnraw().get_nrows(); i++) {
		if (sel.get_line(i)) {
			float v = get_value(i, col);
			if (v > vmax) {
				vmax = v;
				max = i;
			}
			if (v < vmin) {
				vmin = v;
				min = i;
			}		
		}
	}
}

void Picviz::PVPlotted::process_from_mapped(PVMapped* mapped, bool keep_views_info)
{
	set_parent_from_ptr(mapped);

	process_from_parent_mapped(keep_views_info);
}

void Picviz::PVPlotted::process_from_parent_mapped(bool keep_views_info)
{
	// Check parent consistency
	auto mapped = get_parent();

	if (!mapped->is_uptodate()) {
		mapped->process_parent_source();
	}

	create_table();
	if (keep_views_info) {
		process_expanded_selections();
	}
	else {
		_expanded_sels.clear();
	}
	if (!current_view()) {
		PVView_sp view = PVView_p();
		view->init_from_plotted(this, false);
		get_parent<PVSource>()->add_view(view);
	}
	else {
		current_view()->init_from_plotted(this, keep_views_info);
	}
}

bool Picviz::PVPlotted::is_uptodate() const
{
	if (!get_parent()->is_uptodate()) {
		return false;
	}

	return _plotting->is_uptodate();
}


void Picviz::PVPlotted::add_column(PVPlottingProperties const& props)
{
	_plotting->add_column(props);
}

void Picviz::PVPlotted::child_added(PVView& child)
{
	if (!current_view()) {
		get_parent<PVSource>()->select_view(child);
	}
}

void Picviz::PVPlotted::serialize_write(PVCore::PVSerializeObject& so)
{
	data_tree_plotted_t::serialize_write(so);

	so.object("plotting", _plotting, QString(), false, (PVPlotting*) NULL, false);

	so.list("expanded_sels", _expanded_sels, "Expanded selections", (ExpandedSelection*) NULL, QStringList(), true, true);
}

void Picviz::PVPlotted::serialize_read(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t v)
{
	so.object("plotting", _plotting, QString(), false, (PVPlotting*) NULL, false);

	so.list("expanded_sels", _expanded_sels, "Expanded selections", (ExpandedSelection*) NULL, QStringList(), true, true);

	data_tree_plotted_t::serialize_read(so, v);
}

void Picviz::PVPlotted::norm_int_plotted(plotted_table_t const& trans_plotted, uint_plotted_table_t& res, PVCol ncols)
{
	// Here, we make every row starting on a 16-byte boundary
	PVRow nrows = trans_plotted.size()/ncols;
	PVRow nrows_aligned = ((nrows+3)/4)*4;
	size_t dest_size = nrows_aligned*ncols;
	res.reserve(dest_size);
#pragma omp parallel for
	for (PVCol c = 0; c < ncols; c++) {
		for (PVRow r = 0; r < nrows; r++) {
			res[c*nrows_aligned+r] = ((uint32_t) ((double)trans_plotted[c*nrows+r] * (double)UINT_MAX));
		}
		for (PVRow r = nrows; r < nrows_aligned; r++) {
			res[c*nrows_aligned+r] = 0;
		}
	}
}
