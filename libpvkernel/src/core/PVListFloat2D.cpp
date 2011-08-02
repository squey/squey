//! \file PVListFloat2D.cpp
//! $Id: PVListFloat2D.cpp 3090 2011-06-09 04:59:46Z stricaud $
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011

#include <pvcore/PVListFloat2D.h>
#include <stdlib.h>
#include <assert.h>


/******************************************************************************
 *
 *   PVCore::PVListFloat2D:PVListFloat2D:()
 *
 *****************************************************************************/
PVCore::PVListFloat2D::PVListFloat2D(){
	data = NULL;
}

/******************************************************************************
 *
 *   PVCore::PVListFloat2D::PVListFloat2D()
 *
 *****************************************************************************/
PVCore::PVListFloat2D::~PVListFloat2D(){
	free();
}

void PVCore::PVListFloat2D::free()
{
	if (data) {
		PVLOG_INFO("In PVListFloat2D destructor\n");
		::free(data);
		data = NULL;
	}
}




/******************************************************************************
 *
 *   PVCore::PVListFloat2D::count()
 *
 *****************************************************************************/
int PVCore::PVListFloat2D::count(){
	return width*height;
}




/******************************************************************************
 *
 *   PVCore::PVListFloat2D::getData()
 *
 *****************************************************************************/
float* PVCore::PVListFloat2D::getData(){
	return data;
}


/******************************************************************************
 *
 *   PVCore::PVListFloat2D::getRowData()
 *
 *****************************************************************************/
float* PVCore::PVListFloat2D::getRowData(PVRow i){
	return data+i*width;
}



/******************************************************************************
 *
 *   PVCore::PVListFloat2D::getHeight()
 *
 *****************************************************************************/
PVRow PVCore::PVListFloat2D::getHeight(){
	return height;
}




/******************************************************************************
 *
 *   PVCore::PVListFloat2D::getValue()
 *
 *****************************************************************************/
float PVCore::PVListFloat2D::getValue(PVRow i, PVCol j){
	assert(data);
	assert(i < height);
	assert(j < width);

	return data[(i*width)+j];
}




/******************************************************************************
 *
 *   PVCore::PVListFloat2D::getWidth()
 *
 *****************************************************************************/
PVCol PVCore::PVListFloat2D::getWidth(){
	return width;
}

/******************************************************************************
 *
 *   PVCore::PVListFloat2D::reserve()
 *
 *****************************************************************************/
void PVCore::PVListFloat2D::reserve(PVCol width_, PVRow height_){
	width = width_;
	height = height_;
	if (data) {
		data = (float*) realloc(data, width*height*sizeof(float));
	}
	else {
		data = (float*)malloc(width*height*sizeof(float));
	}
	if (data == NULL) {
		PVLOG_ERROR("Unable to initialize memory for a float table !\n");
	}
}


/******************************************************************************
 *
 *   PVCore::PVListFloat2D::setValue()
 *
 *****************************************************************************/
void PVCore::PVListFloat2D::setValue(float value, PVRow i, PVCol j){
	assert(data);
	assert(i < height);
	assert(j < width);

	data[(i*width)+j]=value;
}








