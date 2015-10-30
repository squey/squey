/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

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

	if (initial_row_count <= PICVIZ_INDEX_ARRAY_MAX_SIZE) {
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
