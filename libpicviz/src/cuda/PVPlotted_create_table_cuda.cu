/**
 * \file PVPlotted_create_table_cuda.cu
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#include <picviz/cuda/PVPlotted_create_table_cuda.h>
#include <picviz/cuda/PVCudaDefine.h>






/*******************  prototypes  ***************************************************/
__device__ float device_plotting_integer_port(float value);
__device__ float device_plotting_enum_default(float value);
__device__ float device_plotting_integer_default(float value);
__device__ float device_plotting_ipv4_default(float value);
__device__ float device_plotting_time_24h(float value);
__device__ float device_plotting_time_default(float value);
__device__ float device_plotting_time_week(float value);
__device__ float device_plotting_time_month(float value);
__device__ float device_plotting_integer_minmax(float value, float min, float max);
__device__ float device_plotting_string_default(float value);
/*******************  prototypes  ***************************************************/




/*********************************************************************
*
* Kernel
*
*********************************************************************/
__global__ void PVPlotted_create_table_cuda_kernel(int nrows,int mapped_col_count, float* data_in,float* data_out, PlottingParam *plottingType){
  int idx = blockDim.x*blockIdx.x+threadIdx.x;
  int i = idx/mapped_col_count;
  int j = idx - (mapped_col_count*i);
  //determine which axis it's
  
    if(i < nrows & j< mapped_col_count){//detecte overflow
      
	//value to modify
	float value = data_in[idx];
    
	switch(plottingType[j].type){
	  case time_24h:
	  case time_default:
	    value = device_plotting_time_24h(value);
	    break;
	  case ipv4_default:
	    value = device_plotting_ipv4_default(value);
	    break;
	  case integer_minmax:
	    value = device_plotting_integer_minmax(value, plottingType[j].data[0], plottingType[j].data[1]);
	    break;
	  case integer_default:
	    value = device_plotting_integer_default(value);
	    break;
	  case enum_default:
	    value = device_plotting_enum_default(value);
	    break;
	  case integer_port:
	    value = device_plotting_integer_port(value);
	    break;
	  case string_default:
	    break;
	  default :
	    value = 0.5;
	    break;
	}
	data_in[idx] = value;
    }
  
}
/*********************************************************************
*
* Device function device_plotting_string_default(float value);
*
*********************************************************************/
__device__ float device_plotting_string_default(float value){
  return value;  
}

/*********************************************************************
*
* Device function device_plotting_integer_minmax(float value, float min, float max);
*
*********************************************************************/
__device__ float device_plotting_integer_minmax(float value, float min, float max){
	if (min == max){
	  return 0.5;
	}
	return (value - min) / (max - min);
}

/*********************************************************************
*
* Device function device_plotting_time_24h(float value);
*
*********************************************************************/
__device__ float device_plotting_time_24h(float value){
  return (value / PICVIZ_TIME_24H_MAX);
}

/*********************************************************************
*
* Device function device_plotting_time_default(float value);
*
*********************************************************************/
__device__ float device_plotting_time_default(float value){
  return (value / PICVIZ_TIME_24H_MAX);
}

/*********************************************************************
*
* Device function device_plotting_time_week(float value);
*
*********************************************************************/
__device__ float device_plotting_time_week(float value){
  return (value / PICVIZ_TIME_WEEK_MAX);
}

/*********************************************************************
*
* Device function device_plotting_time_month(float value);
*
*********************************************************************/
__device__ float device_plotting_time_month(float value){
  return (value / PICVIZ_TIME_MONTH_MAX);
}


/*********************************************************************
*
* Device function device_plotting_integer_port(float value);
*
*********************************************************************/
__device__ float device_plotting_integer_port(float value){
  if (value <= 1024) {
    value = ((value - 0.5) / 1024);
  } else {
    value = ((value / (2*65535)) + 0.5);		
  }
  return value;
}

/*********************************************************************
*
* Device function device_plotting_ipv4_default(float value);
*
*********************************************************************/
__device__ float device_plotting_ipv4_default(float value){
  return (value / PICVIZ_IPV4_MAXVAL);
}

/*********************************************************************
*
* Device function device_plotting_integer_default(float value);
*
*********************************************************************/
__device__ float device_plotting_integer_default(float value){
  return (value / 65535);
}

/*********************************************************************
*
* Device function device_plotting_enum_default(float value);
*
*********************************************************************/
__device__ float device_plotting_enum_default(float value){
  return value;
}


/*********************************************************************
*
* Host : Caller
*
*********************************************************************/
 __host__ void PVPlotted_create_table_cuda(int nrows,int mapped_col_count, float* data_h, float* data_return_h, PlottingParam *plottingType)
{
    float *data_d_cu=0;
    float *returned_d_cu=0;
    PlottingParam *plottingType_d_cu=0;
    
    int blockSize = 1024; ///FIXME find in Cuda SDK /opt/cuda_sdk/C/bin/linux/release/deviceQuery how to get max size for grid an block

    
      //config
    dim3 gridSize_cu( ceil(nrows*mapped_col_count/(float)blockSize));
    dim3 blockSize_cu(blockSize);
    

    
    PVLOG_INFO("cuda start malloc :data_d_cu\n");
    //allocate on device 
    cudaMalloc((void**)&data_d_cu, (size_t)nrows * mapped_col_count * sizeof(float));
    PVLOG_INFO("---------------------------------\n");
    check_cuda();
    PVLOG_INFO("cuda start malloc :returned_d_cu\n");
    cudaMalloc((void**)&returned_d_cu, (size_t)nrows * mapped_col_count * sizeof(float));
    check_cuda();
    PVLOG_INFO("cuda start malloc :plottingType_d_cu\n");
    cudaMalloc((void**)&plottingType_d_cu, (size_t)mapped_col_count*sizeof(PlottingParam));
    check_cuda();

    
    //memcpy host->device
    PVLOG_INFO("cuda memcpy 1\n");
    cudaMemcpy(  (float*)data_d_cu,  (float*)data_h,  nrows * mapped_col_count*sizeof(float),  cudaMemcpyHostToDevice);
    check_cuda();
    cudaMemcpy(  (int*)plottingType_d_cu,  (int*)plottingType,  mapped_col_count*sizeof(PlottingParam),  cudaMemcpyHostToDevice);
    check_cuda();

    cudaThreadSynchronize();    
    check_cuda();
    
    //kernel call
    PVLOG_INFO("cuda kernel call\n");
    PVPlotted_create_table_cuda_kernel<<<gridSize_cu,blockSize_cu>>>( nrows, mapped_col_count, data_d_cu, returned_d_cu, plottingType_d_cu );    
    check_cuda();

    //waitting for the end of kernel execution.
    cudaThreadSynchronize();    
    check_cuda();

    
    //memcpy device->host
    PVLOG_INFO("cuda memcpy 2\n");
    //cudaMemcpy(  (float*)data_return_h, (float*) returned_d_cu, nrows * mapped_col_count*sizeof(float),  cudaMemcpyDeviceToHost);    
    cudaMemcpy(  (float*)data_return_h, (float*) data_d_cu, nrows * mapped_col_count*sizeof(float),  cudaMemcpyDeviceToHost);    
    check_cuda();

    
    //free on device
    PVLOG_INFO("cuda free\n");
    cudaFree((void*)data_d_cu);
    check_cuda();
    cudaFree((void*)returned_d_cu);
    check_cuda();
    cudaFree((void*)plottingType_d_cu);
    check_cuda(); 

    PVLOG_INFO("cuda end\n");
}
//mapped_col_count return the number of axis
//nrows number of line log



















