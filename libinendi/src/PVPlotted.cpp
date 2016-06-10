/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <QList>
#include <QStringList>
#include <QString>
#include <QVector>

#include <stdlib.h>
#include <limits>

#include <pvkernel/core/PVMatrix.h>

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

	tbb::tick_count tstart = tbb::tick_count::now();

	// Create our own plugins from the library
	std::vector<PVPlottingFilter::p_type> plotting_filters;
	plotting_filters.resize(mapped_col_count);
	for (PVCol j = 0; j < mapped_col_count; j++) {
		PVPlottingFilter::p_type mf = _plotting.get_filter_for_col(j);
		if (mf) {
			plotting_filters[j] = mf->clone<PVPlottingFilter>();
		}
	}

	_last_updated_cols.clear();
	_minmax_values.resize(mapped_col_count);

	try {

		PVLOG_INFO("(PVPlotted::create_table) begin parallel plotting\n");
		for (PVCol j = 0; j < mapped_col_count; j++) {
			if (_plotting.is_col_uptodate(j)) {
				continue;
			}
			PVPlottingFilter::p_type plotting_filter = plotting_filters[j];
			if (!plotting_filter) {
				PVLOG_ERROR("No valid plotting filter function is defined for axis %d !\n", j);
				continue;
			}

			plotting_filter->set_mapping_mode(get_parent().get_mapping().get_mode_for_col(j));
			plotting_filter->set_mandatory_params(
			    get_parent().get_mapping().get_mandatory_params_for_col(j));
			plotting_filter->set_dest_array(nrows, get_column_pointer(j));
			plotting_filter->set_decimal_type(get_parent().get_decimal_type_of_col(j));
			boost::this_thread::interruption_point();
			tbb::tick_count plstart = tbb::tick_count::now();
			plotting_filter->operator()(get_parent().get_column_pointer(j));
			tbb::tick_count plend = tbb::tick_count::now();

			PVLOG_INFO("(PVPlotted::create_table) parallel plotting for axis %d took "
			           "%0.4f seconds, plugin was %s.\n",
			           j, (plend - plstart).seconds(),
			           qPrintable(plotting_filter->registered_name()));

			boost::this_thread::interruption_point();
			_plotting.set_uptodate_for_col(j);
			_last_updated_cols.push_back(j);

			get_col_minmax(_minmax_values[j].min, _minmax_values[j].max, j);
		}
		PVLOG_INFO("(PVPlotted::create_table) end parallel plotting\n");

		tbb::tick_count tend = tbb::tick_count::now();
		PVLOG_INFO("(PVPlotted::create_table) plotting took %0.4f seconds.\n",
		           (tend - tstart).seconds());
	} catch (boost::thread_interrupted const& e) {
		PVLOG_INFO("(PVPlotted::create_table) plotting canceled.\n");
		throw e;
	}

	return 0;
}

bool Inendi::PVPlotted::dump_buffer_to_file(QString const& file, bool write_as_transposed) const
{
	// Dump the plotted buffer into a file
	// Format is:
	//  * 4 bytes: number of columns
	//  * 1 byte: is it written in a transposed form
	//  * the rest is the plotted

	QFile f(file);
	if (!f.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
		PVLOG_ERROR("Error while opening %s for writing: %s.\n", qPrintable(file),
		            qPrintable(f.errorString()));
		return false;
	}

	PVCol ncols = get_column_count();
	f.write((const char*)&ncols, sizeof(ncols));
	f.write((const char*)&write_as_transposed, sizeof(bool));

	const uint32_t* buf_to_write = get_column_pointer(0);
	PVCore::PVMatrix<uint32_t, PVRow, PVCol> plotted;
	if (!write_as_transposed) {
		PVCore::PVMatrix<uint32_t, PVCol, PVRow> matrix_plotted;
		matrix_plotted.set_raw_buffer((uint32_t*)buf_to_write, get_column_count(),
		                              get_aligned_row_count());
		matrix_plotted.transpose_to(plotted);
		buf_to_write = plotted.get_row_ptr(0);
	}

	const ssize_t sbuf_col = get_row_count() * sizeof(uint32_t);
	for (PVCol j = 0; j < get_column_count(); j++) {
		if (f.write((const char*)get_column_pointer(j), sbuf_col) != sbuf_col) {
			PVLOG_ERROR("Error while writing '%s': %s.\n", qPrintable(file),
			            qPrintable(f.errorString()));
			return false;
		}
	}
	f.close();

	return true;
}

bool Inendi::PVPlotted::load_buffer_from_file(uint_plotted_table_t& buf,
                                              PVRow& nrows,
                                              PVCol& ncols,
                                              bool get_transposed_version,
                                              QString const& file)
{
	ncols = 0;

	FILE* f = fopen(qPrintable(file), "r");
	if (!f) {
		PVLOG_ERROR("Error while opening %s for writing: %s.\n", qPrintable(file), strerror(errno));
		return false;
	}

	static_assert(sizeof(off_t) == sizeof(uint64_t),
	              "sizeof(off_t) != sizeof(uint64_t). Please define "
	              "-D_FILE_OFFSET_BITS=64");

	// Get file size
	fseek(f, 0, SEEK_END);
	const uint64_t fsize = ftello(f);
	fseek(f, 0, SEEK_SET);

	ssize_t size_buf = fsize - sizeof(PVCol) - sizeof(bool);
	if (size_buf <= 0) {
		fclose(f);
		PVLOG_ERROR("File is too small to be valid !\n");
		return false;
	}

	if (fread((void*)&ncols, sizeof(PVCol), 1, f) != 1) {
		PVLOG_ERROR("File is too small to be valid !\n");
		fclose(f);
		return false;
	}
	bool is_transposed = false;
	if (fread((char*)&is_transposed, sizeof(bool), 1, f) != 1) {
		PVLOG_ERROR("Error while reading '%s': %s.\n", qPrintable(file), strerror(errno));
		fclose(f);
		return false;
	}

	bool must_transpose = (is_transposed != get_transposed_version);

	const size_t nuints = size_buf / sizeof(uint32_t);
	nrows = nuints / ncols;
	const size_t nrows_aligned =
	    ((nrows + PVROW_VECTOR_ALIGNEMENT - 1) / (PVROW_VECTOR_ALIGNEMENT)) *
	    PVROW_VECTOR_ALIGNEMENT;
	buf.resize(nrows_aligned * ncols);

	PVLOG_INFO("(Inendi::load_buffer_from_file) number of cols: %d , nuint: %u, "
	           "nrows: %u\n",
	           ncols, nuints, nuints / ncols);

	uint32_t* dest_buf = &buf[0];
	if (must_transpose) {
		dest_buf = (uint32_t*)malloc(nrows_aligned * ncols * sizeof(uint32_t));
	}

	for (PVCol j = 0; j < ncols; j++) {
		if (fread((void*)&dest_buf[j * nrows_aligned], sizeof(uint32_t), nrows, f) != nrows) {
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
		} else {
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

bool Inendi::PVPlotted::load_buffer_from_file(plotted_table_t& buf,
                                              PVCol& ncols,
                                              bool get_transposed_version,
                                              QString const& file)
{
	ncols = 0;

	FILE* f = fopen(qPrintable(file), "r");
	if (!f) {
		PVLOG_ERROR("Error while opening %s for writing: %s.\n", qPrintable(file), strerror(errno));
		return false;
	}

	static_assert(sizeof(off_t) == sizeof(uint64_t),
	              "sizeof(off_t) != sizeof(uint64_t). Please define "
	              "-D_FILE_OFFSET_BITS=64");

	// Get file size
	fseek(f, 0, SEEK_END);
	const uint64_t fsize = ftello(f);
	fseek(f, 0, SEEK_SET);

	ssize_t size_buf = fsize - sizeof(PVCol) - sizeof(bool);
	if (size_buf <= 0) {
		fclose(f);
		PVLOG_ERROR("File is too small to be valid !\n");
		return false;
	}

	if (fread((void*)&ncols, sizeof(PVCol), 1, f) != 1) {
		PVLOG_ERROR("File is too small to be valid !\n");
		fclose(f);
		return false;
	}
	bool is_transposed = false;
	if (fread((char*)&is_transposed, sizeof(bool), 1, f) != 1) {
		PVLOG_ERROR("Error while reading '%s': %s.\n", qPrintable(file), strerror(errno));
		fclose(f);
		return false;
	}

	bool must_transpose = (is_transposed != get_transposed_version);

	size_t nfloats = size_buf / sizeof(float);
	size_t size_read = nfloats * sizeof(float);
	buf.resize(nfloats);

	PVLOG_INFO("(Inendi::load_buffer_from_file) number of cols: %d , nfloats: "
	           "%u, nrows: %u\n",
	           ncols, nfloats, nfloats / ncols);

	float* dest_buf = &buf[0];
	if (must_transpose) {
		dest_buf = (float*)malloc(size_read);
	}

	if (fread((void*)dest_buf, sizeof(float), nfloats, f) != nfloats) {
		PVLOG_ERROR("Error while reading '%s': %s.\n", qPrintable(file), strerror(errno));
		return false;
	}

	if (must_transpose) {
		if (is_transposed) {
			PVCore::PVMatrix<float, PVRow, PVCol> final;
			PVCore::PVMatrix<float, PVCol, PVRow> org;
			org.set_raw_buffer(dest_buf, ncols, nfloats / ncols);

			org.transpose_to(final);
		} else {
			PVCore::PVMatrix<float, PVRow, PVCol> org;
			PVCore::PVMatrix<float, PVCol, PVRow> final;
			org.set_raw_buffer(dest_buf, nfloats / ncols, ncols);
			final.set_raw_buffer(&buf[0], ncols, nfloats / ncols);
			org.transpose_to(final);
		}
		free(dest_buf);
	}

	fclose(f);

	return true;
}

PVRow Inendi::PVPlotted::get_row_count() const
{
	return get_parent<PVSource>().get_row_count();
}

PVCol Inendi::PVPlotted::get_column_count() const
{
	return get_parent<PVMapped>().get_column_count();
}

void Inendi::PVPlotted::to_csv()
{
	PVRow row_count;
	PVCol col_count;

	row_count = get_row_count();
	col_count = get_column_count();

	for (PVRow r = 0; r < row_count; r++) {
		for (PVCol c = 0; c < col_count; c++) {
			printf("%u", get_value(r, c));
			if (c != col_count - 1) {
				std::cout << "|";
			}
		}
		std::cout << "\n";
	}
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

void Inendi::PVPlotted::process_parent_mapped()
{
	create_table();
}

void Inendi::PVPlotted::process_from_parent_mapped()
{
	// Check parent consistency
	auto& mapped = get_parent();

	if (!mapped.is_uptodate()) {
		mapped.compute();
	}

	process_parent_mapped();

	if (get_children().empty()) {
		emplace_add_child();
	}
	for (auto view : get_children()) {
		view->process_parent_plotted();
	}
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

void Inendi::PVPlotted::norm_int_plotted(plotted_table_t const& trans_plotted,
                                         uint_plotted_table_t& res,
                                         PVCol ncols)
{
	// Here, we make every row starting on a 16-byte boundary
	PVRow nrows = trans_plotted.size() / ncols;
	PVRow nrows_aligned = ((nrows + 3) / 4) * 4;
	size_t dest_size = nrows_aligned * ncols;
	res.resize(dest_size);
#pragma omp parallel for
	for (PVCol c = 0; c < ncols; c++) {
		for (PVRow r = 0; r < nrows; r++) {
			res[c * nrows_aligned + r] =
			    ((uint32_t)((double)trans_plotted[c * nrows + r] * (double)PVPlotted::MAX_VALUE));
		}
		for (PVRow r = nrows; r < nrows_aligned; r++) {
			res[c * nrows_aligned + r] = 0;
		}
	}
}
