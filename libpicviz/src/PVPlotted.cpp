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
	remove_all_children();
	PVLOG_DEBUG("In PVPlotted destructor\n");
	for (PVView_sp& v: get_children()) {
		std::cout << v.use_count() << std::endl;
	}
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

	// Transposed normalized unisnged integer.
	// Align the number of lines on a mulitple of 4, in order to have 16-byte aligned starting adresses for each axis
	
	const PVRow nrows_aligned = get_aligned_row_count();
	_uint_table.resize((size_t) mapped_col_count * (size_t) nrows_aligned);
	
	tbb::tick_count tstart = tbb::tick_count::now();
	
	// Create our own plugins from the library
	std::vector<PVPlottingFilter::p_type> plotting_filters;
	plotting_filters.resize(mapped_col_count);
	for (PVCol j = 0; j < mapped_col_count; j++) {
		PVPlottingFilter::p_type mf = _plotting->get_filter_for_col(j);
		if (mf) {
			plotting_filters[j] = mf->clone<PVPlottingFilter>();
		}
	}

	_last_updated_cols.clear();

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
			plotting_filter->set_dest_array(nrows, get_column_pointer(j));
			plotting_filter->set_decimal_type(get_parent()->get_decimal_type_of_col(j));
			boost::this_thread::interruption_point();
			tbb::tick_count plstart = tbb::tick_count::now();
			plotting_filter->operator()(get_parent()->get_column_pointer(j));
			tbb::tick_count plend = tbb::tick_count::now();

			PVLOG_INFO("(PVPlotted::create_table) parallel plotting for axis %d took %0.4f seconds, plugin was %s.\n", j, (plend-plstart).seconds(), qPrintable(plotting_filter->registered_name()));

			boost::this_thread::interruption_point();
			_plotting->set_uptodate_for_col(j);
			_last_updated_cols.push_back(j);
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
	
#if 0
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
#endif
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

	const uint32_t* buf_to_write = get_column_pointer(0);
	PVCore::PVMatrix<uint32_t, PVRow, PVCol> plotted;
	if (!write_as_transposed) {
		PVCore::PVMatrix<uint32_t, PVCol, PVRow> matrix_plotted;
		matrix_plotted.set_raw_buffer((uint32_t*) buf_to_write, get_column_count(), get_aligned_row_count());
		matrix_plotted.transpose_to(plotted);
		buf_to_write = plotted.get_row_ptr(0);
	}
	
	const ssize_t sbuf_col = get_row_count()*sizeof(uint32_t);
	for (PVCol j = 0; j < get_column_count(); j++) {
		if (f.write((const char*) get_column_pointer(j), sbuf_col) != sbuf_col) {
			PVLOG_ERROR("Error while writing '%s': %s.\n", qPrintable(file), qPrintable(f.errorString()));
			return false;
		}
	}
	f.close();

	return true;
}

bool Picviz::PVPlotted::load_buffer_from_file(uint_plotted_table_t& buf, PVRow& nrows, PVCol& ncols, bool get_transposed_version, QString const& file)
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

	const size_t nuints = size_buf/sizeof(uint32_t);
	nrows = nuints/ncols;
	const size_t nrows_aligned = ((nrows+PVROW_VECTOR_ALIGNEMENT-1)/(PVROW_VECTOR_ALIGNEMENT))*PVROW_VECTOR_ALIGNEMENT;
	const size_t size_read_col = nrows*sizeof(uint32_t);
	buf.resize(nrows_aligned*ncols);

	PVLOG_INFO("(Picviz::load_buffer_from_file) number of cols: %d , nuint: %u, nrows: %u\n", ncols, nuints, nuints/ncols);

	uint32_t* dest_buf = &buf[0];
	if (must_transpose) {
		dest_buf = (uint32_t*) malloc(nrows_aligned*ncols*sizeof(uint32_t));
	}

	for (PVCol j = 0; j < ncols; j++) {
		if (fread((void*) &dest_buf[j*nrows_aligned], sizeof(uint32_t), nrows, f) != nrows) {
			PVLOG_ERROR("Error while reading '%s': %s.\n", qPrintable(file), strerror(errno));
			return false;
		}
	}

	if (must_transpose) {
		if (is_transposed) {
			PVCore::PVMatrix<uint32_t, PVCol, PVRow> final;
			PVCore::PVMatrix<uint32_t, PVRow, PVCol> org;
			org.set_raw_buffer(dest_buf, nrows, ncols);
			
			org.transpose_to(final);
		}
		else {
			PVCore::PVMatrix<uint32_t, PVCol, PVRow> org;
			PVCore::PVMatrix<uint32_t, PVRow, PVCol> final;
			org.set_raw_buffer(dest_buf, nrows_aligned, ncols);
			final.set_raw_buffer(&buf[0], ncols, nrows_aligned);
			org.transpose_to(final);
		}
		free(dest_buf);
	}

	fclose(f);

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

void Picviz::PVPlotted::to_csv()
{
	PVRow row_count;
	PVCol col_count;

	row_count = get_row_count();
	col_count = get_column_count();

	for (PVRow r = 0; r < row_count; r++) {
		for (PVCol c = 0; c < col_count; c++) {
			printf("%x", get_value(r,c));
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

QList<PVCol> Picviz::PVPlotted::get_columns_indexes_values_within_range(uint32_t min, uint32_t max, double rate)
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
		const uint32_t* cplotted = get_column_pointer(j);
		for (PVRow i = 0; i < nrows; i++) {
			const uint32_t v = cplotted[i];
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

QList<PVCol> Picviz::PVPlotted::get_columns_indexes_values_not_within_range(uint32_t const min, uint32_t const max, double rate)
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
		const uint32_t* cplotted = get_column_pointer(j);
		for (PVRow i = 0; i < nrows; i++) {
			const uint32_t v = cplotted[i];
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

void Picviz::PVPlotted::get_sub_col_minmax(plotted_sub_col_t& ret, uint32_t& min, uint32_t& max, PVSelection const& sel, PVCol col) const
{
	min = UINT_MAX;
	max = 0;
	const PVRow size = get_row_count();
	ret.reserve(sel.get_number_of_selected_lines_in_range(0, size));
	sel.visit_selected_lines([&](PVRow const r)
		{
			const uint32_t v = this->get_value(r, col);
			if (v > max) {
			   max = v;
			}
			if (v < min) {
				min = v;
			}		
			ret.push_back(plotted_sub_col_t::value_type(r, v));
		},
		size);
}

void Picviz::PVPlotted::get_col_minmax(PVRow& min, PVRow& max, PVSelection const& sel, PVCol col) const
{
	uint32_t vmin,vmax;
	vmin = UINT_MAX;
	vmax = 0;
	min = 0;
	max = 0;
	const PVRow nrows = get_row_count();
	sel.visit_selected_lines([&](PVRow i) {
		const uint32_t v = this->get_value(i, col);
		if (v > vmax) {
			vmax = v;
			max = i;
		}
		if (v < vmin) {
			vmin = v;
			min = i;
		}
	}, nrows);
}

void Picviz::PVPlotted::process_parent_mapped()
{
	create_table();
}

void Picviz::PVPlotted::process_from_parent_mapped()
{
	// Check parent consistency
	auto mapped = get_parent();

	if (!mapped->is_uptodate()) {
		mapped->process_parent_source();
	}

	process_parent_mapped();
	process_expanded_selections();
	
	PVView_sp cur_view;
	if (get_children_count() == 0) {
		cur_view = PVView_p(shared_from_this());
	}
	for (auto view : get_children<PVView>()) {
		view->process_parent_plotted();
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
	get_parent<PVSource>()->add_view(child.shared_from_this());
}

bool Picviz::PVPlotted::is_current_plotted() const
{
	Picviz::PVView const* cur_view = get_parent<PVSource>()->current_view();
	for (auto const& cv: get_children()) {
		if (cv.get() == cur_view) {
			return true;
		}
	}
	return false;
}


QList<PVCol> Picviz::PVPlotted::get_columns_to_update() const
{
	QList<PVCol> ret;

	for (PVCol j = 0; j < get_column_count(); j++) {
		if (!_plotting->is_col_uptodate(j)) {
			ret << j;
		}
	}

	return ret;
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
