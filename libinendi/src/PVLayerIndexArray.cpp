/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <inendi/PVLayerIndexArray.h>




/******************************************************************************
 *
 * Inendi::PVLayerIndexArray::PVLayerIndexArray
 *
 *****************************************************************************/
Inendi::PVLayerIndexArray::PVLayerIndexArray(int initial_row_count)
{
	if ( (0 <= initial_row_count) && (initial_row_count <= INENDI_LAYER_INDEX_ARRAY_MAX_SIZE) ) {
		row_count = initial_row_count;
	} else {
		PVLOG_ERROR("Cannot set row_count while creating PVLayerIndexArray because it is out of range!\n");
		row_count = INENDI_LAYER_INDEX_ARRAY_MAX_SIZE;
	}

}

/******************************************************************************
 *
 * Inendi::PVLayerIndexArray::initialize
 *
 *****************************************************************************/
void Inendi::PVLayerIndexArray::initialize()
{
	int k;

	for (k = 0; k < row_count; k++) {
		array[k] = 0;
	}
}



/******************************************************************************
 *
 * Inendi::PVLayerIndexArray::set_row_count
 *
 *****************************************************************************/
void Inendi::PVLayerIndexArray::set_row_count(int new_row_count)
{
	if ( (0 <= new_row_count) && (new_row_count <= INENDI_LAYER_INDEX_ARRAY_MAX_SIZE) ) {
		row_count = new_row_count;
	} else {
		PVLOG_ERROR("Cannot set_row_count because row_count is out of range!\n");
	}
}

/******************************************************************************
 *
 * Inendi::PVLayerIndexArray::set_value
 *
 *****************************************************************************/
void Inendi::PVLayerIndexArray::set_value(int row_index, int value)
{
	if (( 0 <= row_index) && (row_index < row_count)) {
		array[row_index] = value;
	} else {
		PVLOG_ERROR("Cannot set_value because row_index is out of range!\n");
	}
}

void Inendi::PVLayerIndexArray::serialize(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t /*v*/)
{
	so.buffer("array", &array, sizeof(int)*INENDI_LAYER_INDEX_ARRAY_MAX_SIZE);
	so.attribute("row_count", row_count);
}
