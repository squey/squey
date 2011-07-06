//! \file PVMapped_create_table_cuda.cu
//! $Id: $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <iostream>

#include <picviz/cuda/PVMapped_create_table_cuda.h>
#include <picviz/cuda/PVCudaDefine.h>




struct elt {
    char *str;
    int count;
    elt *next;
};
typedef elt* elt_p;

typedef struct{
  elt* head;
} List;
typedef List* List_p;


/*********************************************************************
*
* __device__ strcmp
*
*********************************************************************/
__device__ bool strcmp(char* str1, char *str2){
    bool isEqual=true;
    int k=0;
    while((str1[k]!=0x00)&&(str2[k]!=0x00)&&(isEqual)){//for each char of string...
	if(str1[k]!=str2[k]){//if one is not equal...
	  return false;
	    isEqual=false;///TODO 
	    break;
	}
	k++;
    }
    return isEqual;
}


/*********************************************************************
*
* __device__ int list_count(List head_list)
*
*********************************************************************/
__device__ int list_count(List_p list){
    elt_p elt = list->head;
    int count=0;
    while(elt->next!=0){
	elt = elt->next;
	count++;
    }
    return count;
}


/*********************************************************************
*
* __device__ List get_end_list(List head_list)
*
*********************************************************************/
__device__ elt_p get_end_list(List_p list){
    elt_p elt = list->head;
    while(elt->next!=0){
	elt = elt->next;
    }
    return elt;
}

/*********************************************************************
*
* __device__ List get_elt(List head_list)
*
*********************************************************************/
__device__ elt_p get_elt(List_p head_list, int k){
  elt_p elt = head_list->head;
  for(int i=0;i<k;i++){
    elt = elt->next;
  }
  return elt;
}

/*********************************************************************
*
* __device__ List get_position
*
*********************************************************************/
__device__ int get_position(List_p list,char* str){
    elt_p elt = list->head;
    int position = 0;
    bool loop=true;
    while(loop){
        if(strcmp(elt->str,str)){
	    return position;
	}
	if(elt->next!=0){
	  elt = elt->next;
	}else{
	  loop=false;
	}
	position++;
    }
    return -1;
}


/*********************************************************************
*
* __device__ add_elt
*
*********************************************************************/
__device__ void add_elt(List_p list,char *str, int* sizeStr){
     if(list->head!=0){//if there is a head , make after last...
	elt_p newElt = (elt_p)malloc(sizeof(elt));
	elt_p last = get_end_list(list);
	last->next = newElt;
	newElt->str = str;
	newElt->count=1;
	newElt->next = 0;
	last = get_end_list(list);
	printf("l");
    }else{//the list is empty, make head...
	printf("h");
	list->head = (elt_p)malloc(sizeof(elt));
	list->head->str = str;
	list->head->count = 1;
	list->head->next = 0;
    }
    printf("%s added in block %d\n",str,blockIdx.x);
    atomicAdd(sizeStr,1);//Use sizeStr (not list.size) let us use atomic op, cause other thread can't use it.
}


/*********************************************************************
*
* __device__ int contains(List head_list)
*
*********************************************************************/
__device__ int contains(List_p hash_head, int max, char *str_query){
    elt_p hash = hash_head->head;
    int position=0;
    bool continueLoop=false;
    if(hash!=0) {//if there is an element...
	do{
	    continueLoop=false;

	    if(strcmp(hash->str,str_query)){//if strings are equal...
	      return position;//return position.
	    }
	    
	    position++;
	    if(hash->next!=0){
	        hash=hash->next;
		continueLoop=true;
	    }
	}while(continueLoop);
    }else{
	return -2;
    }
  
    return -1;
}



/*********************************************************************
*
* __device__ int contains(List head_list)
*
*********************************************************************/
__device__ int getIdx(){
  return blockDim.x*blockIdx.x+threadIdx.x;
}



/*********************************************************************
*
* Kernel 		pvmapped_create_table_enum_default_kernel
*
*********************************************************************/
__global__ void pvmapped_create_table_enum_default_kernel(int nbr_str, char *str, int *size_hash, List_p head_list, int *mutex){

    //__shared__ int mutex;
    //int idx = blockDim.x*blockIdx.x+threadIdx.x;
    int idx = getIdx();
    

    if(idx==0){
      *size_hash=0;
      atomicExch(mutex,0);
    }
    __syncthreads();
    //***************** end init **************


    if(idx<nbr_str-1){
	int position = -0.1;
	    position = contains(head_list,nbr_str,&str[1000*idx]);
	    if(position<0){
	      //In the two folowing line, it seems we do two times the same thing, but in fact we don't go there anytime. So we don't bloc all threads with an atomic any time, to save time.
		atomic_exec(mutex,
		    position = contains(head_list,nbr_str,&str[1000*idx]);
		    if(position<0){
			printf("add needed pos=%d  ",position);
			add_elt(head_list,&str[1000*idx],size_hash);
		    }
		)
	    }else{
		elt_p elt = get_elt(head_list,position);
		atomicAdd(&elt->count,1);
	    }
	
    }
    
    
    //*************** end process ***************
    /*__syncthreads();
    if(idx==0){
      //*size_hash = head_list.size;
      printf("list enum (%)\n",size_hash);
      int nbr=0;
      for(int i=0;i<*size_hash;i++){
	elt_p element = get_elt(head_list,i);
	if(element==0){
	  //printf("error in kernel i=%d\n",i);
	}else{
	    //printf("     %s %d\n",element->str,element->count);
	    nbr++;
	}
      }
      *size_hash=nbr;
    }*/
}


/*********************************************************************
*
* Kernel 		pvmapped_create_table_enum_default_make_result
*
*********************************************************************/
__global__ void pvmapped_create_table_enum_default_make_result(int size_str, List_p list, int *size_list, char *cuda_str_d, float* rst){
    //int idx = blockDim.x*blockIdx.x+threadIdx.x;
    int idx = getIdx();
    *size_list = list_count(list);
    if(idx<size_str){
	int pos = get_position(list,&cuda_str_d[idx*1000]);
	if(pos>=0){
	    rst[idx]=(float)pos/(float)*size_list;
	}else{
	    printf("err %s  %d \n",&cuda_str_d[idx*1000],pos);
	}
    }
}


/*********************************************************************
*
* Host : initCaller
*
*********************************************************************/
void inline initCaller(char *cuda_str_d,   size_t size_of_str,  List_p myEnum,  int *size_hash_d){
   
}


/*********************************************************************
*
* Host : Caller
*
*********************************************************************/
__host__ void pvmapped_create_table_enum_default(int j, char* cuda_str_h, int size_cuda_str_h, PVCore::PVListFloat2D* rst){
    PVLOG_INFO("----------------------------------call pvmapped_create_table_enum_default\n");
    PVLOG_INFO("PICVIZ_AXES_MAX %d \n",PICVIZ_AXES_MAX);
    PVLOG_INFO("size_cuda_str_h %d \n",size_cuda_str_h);
    
    
    
    char *cuda_str_d=0;
    float *returnTable_d=0,*returnTable_h=(float*)malloc(size_cuda_str_h*1000*sizeof(char));
    int *size_hash_d,size_hash_h=0;
    int *mutex_d;
    size_t size_of_str = 1000*size_cuda_str_h;
    List_p myEnum=0;
    dim3 gridSize_cu, blockSize_cu;
    uint blockSize;
    //dim3 
    
    //get cuda properties
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    //set the number max of threads per block read the device properties.
    blockSize = deviceProp.maxThreadsPerBlock;
    
    cudaSetDevice( 0 );
    check_cuda();
  
    
    //initCaller(cuda_str_d,size_of_str,myEnum,size_hash_d);
    
     //allocate
    PVLOG_INFO("cuda start malloc :\n");
    cudaMalloc((void**)&cuda_str_d,size_of_str*sizeof(char));
    check_cuda();
    cudaMemset(cuda_str_d,0,size_of_str*sizeof(char));
    check_cuda();
    cudaMalloc((void**)&myEnum,sizeof(List));
    check_cuda();
    cudaMemset(myEnum,0,sizeof(List));
    check_cuda();
    cudaMalloc((void**)&size_hash_d,sizeof(int));
    check_cuda();
    cudaMalloc((void**)&mutex_d,sizeof(int));
    check_cuda();
    cudaMemset(mutex_d,0,sizeof(int));
    check_cuda();
    
    //mem copy
    PVLOG_INFO("cuda start memcpy :cuda_str_d\n");
    cudaMemcpy(  (char*)cuda_str_d,  (char*)cuda_str_h,  size_of_str*sizeof(char),  cudaMemcpyHostToDevice);
    check_cuda();
    PVLOG_INFO("cuda start memcpy :size_hash_d\n");
    cudaMemcpy(  (int*)size_hash_d,  (int*)&size_hash_h,  1*sizeof(int),  cudaMemcpyHostToDevice);
    check_cuda();
    
    

    
    //kernel call for process
    gridSize_cu=dim3(ceil(size_cuda_str_h/(float)blockSize));
    blockSize_cu=dim3(blockSize);
    PVLOG_INFO("cuda call :grid%d block%d\n",gridSize_cu.x,blockSize_cu.x);
    pvmapped_create_table_enum_default_kernel<<<gridSize_cu,blockSize_cu>>>(size_cuda_str_h,cuda_str_d,size_hash_d,myEnum,mutex_d);
    check_cuda();
    
    
    //get the size needed for the return table
    cudaMemcpy(  (int*)&size_hash_h,  (int*)size_hash_d,  1*sizeof(int),  cudaMemcpyDeviceToHost);
    check_cuda();
    
    //allocate the return table
    ///FIXME Because of impossibility to have atomic function, the is more allocation that we need.
    //PVLOG_INFO("malloc returnTable_d %d\n",size_hash_h);
    cudaMalloc((void**)&returnTable_d,size_cuda_str_h*sizeof(float));
    check_cuda();
    
    //kernel format result
    PVLOG_INFO("cuda call make_result\n");
    pvmapped_create_table_enum_default_make_result<<<gridSize_cu,blockSize_cu>>>(size_cuda_str_h,myEnum,size_hash_d,cuda_str_d,returnTable_d);
    check_cuda();
    //now size_hash_d is the correct count.
    cudaMemcpy(  (int*)&size_hash_h,  (int*)size_hash_d,  1*sizeof(int),  cudaMemcpyDeviceToHost);
    check_cuda();
    
    
    PVLOG_INFO("waitting end of call :\n");
    cudaThreadSynchronize();    
    check_cuda();
    
    //mem copy
    PVLOG_INFO("cuda end memcpy :\n");
    cudaMemcpy((float*)returnTable_h, (float*)returnTable_d, size_cuda_str_h*sizeof(float), cudaMemcpyDeviceToHost);
    check_cuda();
    
    //free
    PVLOG_INFO("cuda free mem :\n");
    cudaFree((void*)cuda_str_d);
    check_cuda();
    cudaFree((void*)size_hash_d);
    check_cuda();
    cudaFree((void*)mutex_d);
    check_cuda();
    
    
    for(int i=0;i<size_cuda_str_h;i++){
      if(size_hash_h>=1){
	rst->setValue(returnTable_h[i],i,j);
      }else{
	rst->setValue(0.5,i,j);
      }
	//PVLOG_WARN("%s val %f\n",&cuda_str_h[1000*i],returnTable_h[i]);
    }
    
    //to make sure we will have a clean device during next use.
    cudaDeviceReset();
    
    PVLOG_INFO("size_hash_h %d at the end \n",size_hash_h);
    PVLOG_INFO("----------------------------------end pvmapped_create_table_enum_default\n");
}
