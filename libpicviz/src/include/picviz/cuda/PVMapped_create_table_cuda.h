/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

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
