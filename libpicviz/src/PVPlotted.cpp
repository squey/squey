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
	#ifndef CUDA
	create_table();
	#else //CUDA
	create_table_cuda();
	#endif //CUDA

	_view.reset(new PVView(this));
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
}

#ifndef CUDA
int PVPlotted::create_table()
{
	PVCol mapped_col_count = _mapped->get_column_count();

	QMutex lock;
	const PVRow nrows = (PVRow)_mapped->table.getHeight();
	
	_table.clear();
	_table.reserve(mapped_col_count * nrows);
	
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

	PVCore::PVListFloat2D trans_table;
	trans_table.reserve(nrows, mapped_col_count);

	PVLOG_INFO("(PVPlotted::create_table) begin parallel plotting\n");
	for (PVCol j = 0; j < mapped_col_count; j++) {
		PVPlottingFilter::p_type plotting_filter = plotting_filters[j];
		if (!plotting_filter) {
			PVLOG_ERROR("No valid plotting filter function is defined for axis %d !\n", j);
			continue;
		}

		plotting_filter->set_mapping_mode(_plotting.get_format()->get_axes().at(j).get_mapping());
		plotting_filter->set_mandatory_params(_mapped->get_mapping().get_mandatory_params_for_col(j));
		plotting_filter->set_dest_array(nrows, trans_table.getRowData(j));
		tbb::tick_count plstart = tbb::tick_count::now();
		plotting_filter->operator()(_mapped->trans_table.getRowData(j));
		tbb::tick_count plend = tbb::tick_count::now();
		int64_t nrows_tmp = nrows;

		PVLOG_INFO("(PVPlotted::create_table) parallel plotting for axis %d took %0.4f seconds.\n", j, (plend-plstart).seconds());

		// TODO: this is a matrix transposition. Find out the best way to do this !
#pragma omp parallel for
		for (int64_t i = 0; i < nrows_tmp; i++) {
			float v = trans_table.getValue(j, i);
			_table[i*mapped_col_count+j] = v;
#ifndef NDEBUG
			// Check that every plotted value is between 0 and 1
			if (v > 1 || v < 0) {
				PVLOG_WARN("Plotting value for row/col %d/%d is %0.4f !\n", i,j,v);
			}
#endif
		}
	}
	PVLOG_INFO("(PVPlotted::create_table) end parallel plotting\n");

	tbb::tick_count tend = tbb::tick_count::now();
	PVLOG_INFO("(PVPlotted::create_table) plotting took %0.4f seconds.\n", (tend-tstart).seconds());

	// Free the table of the PVMapped object
	_mapped->trans_table.free();

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
	return _mapped->table.getHeight();
}

PVCol PVPlotted::get_column_count() const
{
	return _mapped->table.getWidth();
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
				std::cout << ",";
			}
		}
		std::cout << "\n";
	}

}

void Picviz::PVPlotted::get_sub_col_minmax(plotted_sub_col_t& ret, float& min, float& max, PVSelection const& sel, PVCol col) const
{
	min = FLT_MAX;
	max = 0;
	PVRow size = get_qtnraw().size();
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

void Picviz::PVPlotted::process_from_mapped(PVMapped* mapped, bool keep_views_info)
{
	_plotting.set_mapped(mapped);
	set_plotting(_plotting);
	create_table();
	if (!keep_views_info || !_view) {
		_view.reset(new PVView(this));
	}
	else {
		_view->init_from_plotted(this, keep_views_info);
	}
}

void Picviz::PVPlotted::serialize(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t /*v*/)
{
	so.object("plotting", _plotting);
	so.object("view", _view);
}

}
