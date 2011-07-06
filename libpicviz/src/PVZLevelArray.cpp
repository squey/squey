//! \file PVZLevelArray.cpp
//! $Id: PVZLevelArray.cpp 3090 2011-06-09 04:59:46Z stricaud $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011


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