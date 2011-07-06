//! \file PVIndexArray.cpp
//! $Id: PVIndexArray.cpp 3090 2011-06-09 04:59:46Z stricaud $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011


#include <picviz/PVIndexArray.h>




/******************************************************************************
 *
 * Picviz::PVIndexArray::PVIndexArray
 *
 *****************************************************************************/
Picviz::PVIndexArray::PVIndexArray()
{
	index_count = 0;
	row_count = PICVIZ_INDEX_ARRAY_MAX_SIZE;
}

/******************************************************************************
 *
 * Picviz::PVIndexArray::PVIndexArray
 *
 *****************************************************************************/
Picviz::PVIndexArray::PVIndexArray(PVRow initial_row_count)
{
	index_count = 0;

	if ( (0 <= initial_row_count) && (initial_row_count <= PICVIZ_INDEX_ARRAY_MAX_SIZE) ) {
		row_count = initial_row_count;
	} else {
		PVLOG_ERROR("Cannot set row_count while creating picviz_index_array because it is out of range!\n");
		row_count = PICVIZ_INDEX_ARRAY_MAX_SIZE;
	}
}


/******************************************************************************
 *
 * Picviz::PVIndexArray::set_from_selection
 *
 *****************************************************************************/
void Picviz::PVIndexArray::set_from_selection(const PVSelection & selection_)
{
	int index = 0;
	int k;

	for ( k=0; k<row_count; k++) {
		if (selection_.get_line(k)) {
			array[index] = k;
			index++;
		}
	}

	index_count = index;
}
