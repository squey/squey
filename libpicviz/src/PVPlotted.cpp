//! \file PVPlotted.cpp
//! $Id: PVPlotted.cpp 3221 2011-06-30 11:45:19Z aguinet $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

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

namespace Picviz {

PVPlotted::PVPlotted(PVPlotting const& plotting)
{
	set_plotting(plotting);
	process_from_parent_mapped(false);
}

PVPlotted::~PVPlotted()
{
	PVLOG_INFO("In PVPlotted destructor\n");
}

void PVPlotted::set_plotting(PVPlotting const& plotting)
{
	_plotting = plotting;
	root = _plotting.get_root_parent();
	_mapped = _plotting.get_mapped_parent();
	if (_view) {
		get_source_parent()->add_view(_view);
	}
}

#ifndef CUDA
int PVPlotted::create_table()
{
	PVCol mapped_col_count = _mapped->get_column_count();

	const PVRow nrows = (PVRow)_mapped->get_row_count();

	// That buffer will be about 3.8MB for 10 million lines, so we keep it
	// to save some allocations.
	_tmp_values.reserve(nrows);
	float* p_tmp_v = &_tmp_values[0];
	
	_table.resize(mapped_col_count * nrows);
	
	tbb::tick_count tstart = tbb::tick_count::now();
	
	// We will use the trans_table of PVMapped and write "in live" in
	// the "table" of PVPlotted
	
	// Create our own plugins from the library
	std::vector<PVPlottingFilter::p_type> plotting_filters;
	plotting_filters.resize(mapped_col_count);
	for (PVCol j = 0; j < mapped_col_count; j++) {
		PVPlottingFilter::p_type mf = _plotting.get_filter_for_col(j);
		if (mf) {
			plotting_filters[j] = mf->clone<PVPlottingFilter>();
		}
	}

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

			plotting_filter->set_mapping_mode(_mapped->get_mapping().get_mode_for_col(j));
			plotting_filter->set_mandatory_params(_mapped->get_mapping().get_mandatory_params_for_col(j));
			plotting_filter->set_dest_array(nrows, p_tmp_v);
			boost::this_thread::interruption_point();
			tbb::tick_count plstart = tbb::tick_count::now();
			plotting_filter->operator()(_mapped->trans_table.getRowData(j));
			tbb::tick_count plend = tbb::tick_count::now();
			int64_t nrows_tmp = nrows;

			PVLOG_INFO("(PVPlotted::create_table) parallel plotting for axis %d took %0.4f seconds, plugin was %s.\n", j, (plend-plstart).seconds(), qPrintable(plotting_filter->registered_name()));

			boost::this_thread::interruption_point();
//#pragma omp parallel for
			for (int64_t i = 0; i < nrows_tmp; i++) {
				float v = p_tmp_v[i];
				_table[i*mapped_col_count+j] = v;
#ifndef NDEBUG
				// Check that every plotted value is between 0 and 1
				if (v > 1 || v < 0) {
					PVLOG_WARN("Plotting value for row/col %d/%d is %0.4f !\n", i,j,v);
				}
#endif
			}

			_plotting.set_uptodate_for_col(j);
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
#else //CUDA
/***************** CUDA *****************************************/
int PVPlotted::create_table_cuda(){
  
  PVCol mapped_col_count = _mapped->get_column_count();
  QMutex lock;
  const PVRow nrows = (PVRow)_mapped->table.getHeight();
  //float dataTmp[nrows][mapped_col_count];
  
  
  //init table type of profile
  /// TODO do it differently : create a method in PVPlottingPorperties or PVPlottingFunction to generate it
  //for apache
  /*PlottingParam *plottingType = new PlottingParam[9];
  /// FIXME don't feed it hardly
  plottingType[0].type = time_default;
  plottingType[1].type = ipv4_default;
  plottingType[2].type = ipv4_default;
  plottingType[3].type = integer_minmax;
  plottingType[3].data[0] = _mapped->mapping->dict_float[3]["ymin"];
  plottingType[3].data[1] = _mapped->mapping->dict_float[3]["ymax"];
  plottingType[4].type = integer_minmax;
  plottingType[4].data[0] = _mapped->mapping->dict_float[4]["ymin"];
  plottingType[4].data[1] = _mapped->mapping->dict_float[4]["ymax"];
  plottingType[5].type = integer_default;
  plottingType[6].type = enum_default;
  plottingType[7].type = integer_port;
  plottingType[8].type = integer_port;//for apache*/
  
  //for squid
  PlottingParam *plottingType = new PlottingParam[9];
  /// FIXME don't feed it hardly
  plottingType[0].type = time_default;
  plottingType[1].type = enum_default;
  plottingType[2].type = enum_default;
  plottingType[3].type = string_default;
  plottingType[4].type = enum_default;
  plottingType[5].type = enum_default;
  plottingType[6].type = ipv4_default;
  
   
  
  //kernel caller
  ///FIXME hard cast float* (don't use Qt class in cuda caller)
  PVLOG_INFO("cuda call\n");
  PVPlotted_create_table_cuda((int)nrows,(int)mapped_col_count, (float*)_mapped->table.getData(),(float*)&table[0], plottingType);  
  PVLOG_INFO("cuda end\n");
  
  //to see first log
  /*for(int i=0;i<9;i++){
	std::cout<<"*** CUDA TRACE *** cuda val  "<<table[i]<<std::endl;
  }
  std::cout<<"*** CUDA TRACE ***    "<<std::endl;
  for(int i=9;i<18;i++){
	std::cout<<"*** CUDA TRACE *** cuda val  "<<table[i]<<std::endl;
  }
  std::cout<<"*** CUDA TRACE ***    "<<std::endl;*/

  
  
  //old
  /*PVLOG_INFO("linear call\n");
	for (PVRow i = 0; i < nrows; i++) {
		for (PVCol j = 0; j < mapped_col_count; j++) {
			const float val = plotting->get_position(j, _mapped->table.getValue(i,j));
			//lock.lock();
			table[i*mapped_col_count+j] = val;
			//to see first log
			if(i*mapped_col_count+j<18){
			  std::cout<<"*** CUDA TRACE *** no cuda val  "<<val<<" ? "<<table[i*mapped_col_count+j]<<std::endl;
			}
			//lock.unlock();
		}
	}
	PVLOG_INFO("linear end\n");*/
  
  
  return 0;
}
/***************** CUDA *****************************************/
#endif//CUDA

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
	QString plugin = _plotting.get_properties_for_col(axis_id).get_type() + "_" + mode;
	PVPlottingFilter::p_type filter = LIB_CLASS(PVPlottingFilter)::get().get_class_by_name(plugin);
	if (!filter) {
		return;
	}
	PVPlottingFilter::p_type filter_clone = filter->clone<PVPlottingFilter>();

	PVCol ncols = get_column_count();
	assert(axis_id < ncols);
	PVMapped::mapped_sub_col_t sub_plotted;
	float min,max;
	get_mapped_parent()->get_sub_col_minmax(sub_plotted, min, max, sel, axis_id);
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

	QFile f(file);
	if (!f.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
		PVLOG_ERROR("Error while opening %s for writing: %s.\n", qPrintable(file), qPrintable(f.errorString()));
		return false;
	}

	ssize_t size_buf = f.size()-sizeof(PVCol)-sizeof(bool);
	if (size_buf <= 0) {
		PVLOG_ERROR("File is too small to be valid !\n");
		return false;
	}

	f.read((char*) &ncols, sizeof(PVCol));
	bool is_transposed = false;
	if (f.read((char*) &is_transposed, sizeof(bool)) != sizeof(bool)) {
		PVLOG_ERROR("Error while reading '%s': %s.\n", qPrintable(file), qPrintable(f.errorString()));
		return false;
	}

	bool must_transpose = (is_transposed != get_transposed_version);

	ssize_t nfloats = size_buf/sizeof(float);
	ssize_t size_read = nfloats*sizeof(float);
	buf.resize(nfloats);

	float* dest_buf = &buf[0];
	if (must_transpose) {
		dest_buf = (float*) malloc(size_read);
	}

	if (f.read((char*) dest_buf, size_read) != size_read) {
		PVLOG_ERROR("Error while reading '%s': %s.\n", qPrintable(file), qPrintable(f.errorString()));
		return false;
	}

	if (must_transpose) {
		if (is_transposed) {
			PVCore::PVMatrix<float, PVRow, PVCol> final;
			PVCore::PVMatrix<float, PVCol, PVRow> org;
			org.set_raw_buffer(dest_buf, ncols, nfloats/ncols);
			final.set_raw_buffer(&buf[0], nfloats/ncols, ncols);
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

	return true;
}

PVRush::PVNraw::nraw_table& PVPlotted::get_qtnraw()
{
	return _plotting.get_qtnraw();
}

const PVRush::PVNraw::nraw_table& PVPlotted::get_qtnraw() const
{
	return _plotting.get_qtnraw();
}

PVRow PVPlotted::get_row_count() const
{
	return _mapped->get_row_count();
}

PVCol PVPlotted::get_column_count() const
{
	return _mapped->get_column_count();
}

PVSource* PVPlotted::get_source_parent()
{
	return _plotting.get_source_parent();
}

float PVPlotted::get_value(PVRow row, PVCol col) const
{
	return _table[row * get_column_count() + col];
}

void PVPlotted::to_csv()
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
	_plotting.set_mapped(mapped);
	set_plotting(_plotting);
	process_from_parent_mapped(keep_views_info);
}

void Picviz::PVPlotted::process_from_parent_mapped(bool keep_views_info)
{
	// Check parent consistency
	if (!_mapped->is_uptodate()) {
		_mapped->process_parent_source();
	}

	create_table();
	if (keep_views_info) {
		process_expanded_selections();
	}
	else {
		_expanded_sels.clear();
	}
	if (!_view) {
		_view.reset(new PVView(this));
		get_source_parent()->add_view(_view);
	}
	else {
		_view->init_from_plotted(this, keep_views_info);
	}
}

bool Picviz::PVPlotted::is_uptodate() const
{
	if (!_mapped->is_uptodate()) {
		return false;
	}

	return _plotting.is_uptodate();
}


void Picviz::PVPlotted::add_column(PVPlottingProperties const& props)
{
	_plotting.add_column(props);
}

void Picviz::PVPlotted::serialize(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t /*v*/)
{
	so.object("plotting", _plotting, QString(), false, (PVPlotting*) NULL, false);
	so.object("view", _view, QObject::tr("View"));

	so.list("expanded_sels", _expanded_sels, "Expanded selections", (ExpandedSelection*) NULL, QStringList(), true, true);
}

}
