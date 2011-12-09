//! \file PVListFloat2D.cpp
//! $Id: PVListFloat2D.cpp 3090 2011-06-09 04:59:46Z stricaud $
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011

#include <pvkernel/core/PVListFloat2D.h>
#include <stdlib.h>
#include <assert.h>


/******************************************************************************
 *
 *   PVCore::PVListFloat2D:PVListFloat2D:()
 *
 *****************************************************************************/
PVCore::PVListFloat2D::PVListFloat2D()
{
	width = 0;
	height = 0;
}

/******************************************************************************
 *
 *   PVCore::PVListFloat2D::PVListFloat2D()
 *
 *****************************************************************************/
PVCore::PVListFloat2D::~PVListFloat2D()
{
	free();
}

void PVCore::PVListFloat2D::free()
{
	data.clear();
	width = 0;
	height = 0;
}




/******************************************************************************
 *
 *   PVCore::PVListFloat2D::count()
 *
 *****************************************************************************/
int PVCore::PVListFloat2D::count() const
{
	return width*height;
}




/******************************************************************************
 *
 *   PVCore::PVListFloat2D::getData()
 *
 *****************************************************************************/
float* PVCore::PVListFloat2D::getData()
{
	return &data[0];
}

const float* PVCore::PVListFloat2D::getData() const
{
	return &data.at(0);
}


/******************************************************************************
 *
 *   PVCore::PVListFloat2D::getRowData()
 *
 *****************************************************************************/
float* PVCore::PVListFloat2D::getRowData(PVRow i)
{
	return &data[i*width];
}

const float* PVCore::PVListFloat2D::getRowData(PVRow i) const
{
	return &data.at(i*width);
}



/******************************************************************************
 *
 *   PVCore::PVListFloat2D::getHeight()
 *
 *****************************************************************************/
PVRow PVCore::PVListFloat2D::getHeight() const
{
	return height;
}




/******************************************************************************
 *
 *   PVCore::PVListFloat2D::getValue()
 *
 *****************************************************************************/
float PVCore::PVListFloat2D::getValue(PVRow i, PVCol j) const
{
	assert(i < height);
	assert(j < width);

	return data.at((i*width)+j);
}




/******************************************************************************
 *
 *   PVCore::PVListFloat2D::getWidth()
 *
 *****************************************************************************/
PVCol PVCore::PVListFloat2D::getWidth() const
{
	return width;
}

/******************************************************************************
 *
 *   PVCore::PVListFloat2D::reserve()
 *
 *****************************************************************************/
bool PVCore::PVListFloat2D::reserve(PVCol width_, PVRow height_)
{
	width = width_;
	height = height_;
	try {
		data.resize(width*height);
	}
	catch (std::bad_alloc const&)
	{
		PVLOG_ERROR("(PVCore::PVListFloat2D::reserve) unable to allocate memory (%ldx%ldx%ld = %ld)!\n", width_, height_, sizeof(float), width_*height_*sizeof(float));
		return false;
	}

	return true;
}


/******************************************************************************
 *
 *   PVCore::PVListFloat2D::setValue()
 *
 *****************************************************************************/
void PVCore::PVListFloat2D::setValue(float value, PVRow i, PVCol j)
{
	assert(i < height);
	assert(j < width);

	data[(i*width)+j]=value;
}








