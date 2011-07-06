//! \file PVPlotted_create_table_cuda.h
//! $Id: $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011


#ifndef PICVIZ_PVPLOTTED_CREATE_TABLE_CUDA_H
#define PICVIZ_PVPLOTTED_CREATE_TABLE_CUDA_H
#ifdef CUDA

//#include <cuda.h>
#include <pvcore/PVLogger.h>

///TODO declare it to accessible everywhere in libpicviz
enum TypeOfPlotting{time_default, time_24h, time_week, time_month, ipv4_default, integer_minmax, integer_default, enum_default, integer_port, string_default};
typedef struct{
  TypeOfPlotting type;
  float data[3];
}PlottingParam;

void PVPlotted_create_table_cuda(int nrows,int mapped_col_count, float* data, float* data_return_h, PlottingParam *plottingType);


#endif /* CUDA */
#endif	/* PICVIZ_PVPLOTTED_CREATE_TABLE_CUDA_H */




