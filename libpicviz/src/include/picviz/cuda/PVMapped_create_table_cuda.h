//! \file PVMapped_create_table_cuda.h
//! $Id: $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011


#ifndef PICVIZ_PVMAPPED_CREATE_TABLE_CUDA_H
#define PICVIZ_PVMAPPED_CREATE_TABLE_CUDA_H
#ifdef CUDA

//#include <cuda.h>
#include <pvkernel/core/PVLogger.h>
#include <pvkernel/core/PVListFloat2D.h>


/**
*@param j column to process
*@param cuda_host_str data to process
*@param rst the result
*/
void pvmapped_create_table_enum_default(int j, char* cuda_host_str, int size_cuda_host_str, PVCore::PVListFloat2D* rst);



#endif /* CUDA */
#endif//PICVIZ_PVMAPPED_CREATE_TABLE_CUDA_H
