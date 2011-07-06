//! \file PVLayerIndexArray.cpp
//! $Id: PVLayerIndexArray.cpp 3090 2011-06-09 04:59:46Z stricaud $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011


#include <picviz/PVLayerIndexArray.h>




/******************************************************************************
 *
 * Picviz::PVLayerIndexArray::PVLayerIndexArray
 *
 *****************************************************************************/
Picviz::PVLayerIndexArray::PVLayerIndexArray(int initial_row_count)
{
	if ( (0 <= initial_row_count) && (initial_row_count <= PICVIZ_LAYER_INDEX_ARRAY_MAX_SIZE) ) {
		row_count = initial_row_count;
	} else {
		PVLOG_ERROR("Cannot set row_count while creating PVLayerIndexArray because it is out of range!\n");
		row_count = PICVIZ_LAYER_INDEX_ARRAY_MAX_SIZE;
	}

}

/******************************************************************************
 *
 * Picviz::PVLayerIndexArray::initialize
 *
 *****************************************************************************/
void Picviz::PVLayerIndexArray::initialize()
{
	int k;

	for (k = 0; k < row_count; k++) {
		array[k] = 0;
	}
}



/******************************************************************************
 *
 * Picviz::PVLayerIndexArray::set_row_count
 *
 *****************************************************************************/
void Picviz::PVLayerIndexArray::set_row_count(int new_row_count)
{
	if ( (0 <= new_row_count) && (new_row_count <= PICVIZ_LAYER_INDEX_ARRAY_MAX_SIZE) ) {
		row_count = new_row_count;
	} else {
		PVLOG_ERROR("Cannot set_row_count because row_count is out of range!\n");
	}
}

/******************************************************************************
 *
 * Picviz::PVLayerIndexArray::set_value
 *
 *****************************************************************************/
void Picviz::PVLayerIndexArray::set_value(int row_index, int value)
{
	if (( 0 <= row_index) && (row_index < row_count)) {
		array[row_index] = value;
	} else {
		PVLOG_ERROR("Cannot set_value because row_index is out of range!\n");
	}
}


