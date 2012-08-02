/**
 * \file PVZLevelArray.cpp
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#include <picviz/PVZLevelArray.h>




/******************************************************************************
 *
 * Picviz::PVZLevelArray::PVZLevelArray
 *
 *****************************************************************************/
Picviz::PVZLevelArray::PVZLevelArray(PVRow initial_row_count)
{
	if ( (0 <= initial_row_count) && (initial_row_count <= PICVIZ_Z_LEVEL_ARRAY_MAX_SIZE) ) {
		row_count = initial_row_count;
	} else {
		PVLOG_ERROR("Cannot set row_count while creating PVZLevelArray because it is out of range!\n");
	}

}

/******************************************************************************
 *
 * Picviz::PVZLevelArray::set_level
 *
 *****************************************************************************/
void Picviz::PVZLevelArray::set_level(PVRow row_index, float level)
{
	array[row_index] = level;
}