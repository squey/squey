/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <inendi/PVLayerIndexArray.h>

#include <pvkernel/core/PVSerializeObject.h>

/******************************************************************************
 *
 * Inendi::PVLayerIndexArray::initialize
 *
 *****************************************************************************/
void Inendi::PVLayerIndexArray::initialize()
{
	std::fill(_array.begin(), _array.end(), 0);
}

/******************************************************************************
 *
 * Inendi::PVLayerIndexArray::set_row_count
 *
 *****************************************************************************/
void Inendi::PVLayerIndexArray::set_row_count(PVRow row_count)
{
	_array.resize(row_count);
}

/******************************************************************************
 *
 * Inendi::PVLayerIndexArray::set_value
 *
 *****************************************************************************/
void Inendi::PVLayerIndexArray::set_value(PVRow row_index, int value)
{
	_array[row_index] = value;
}

void Inendi::PVLayerIndexArray::serialize(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t /*v*/)
{
	PVRow row_count = _array.size();
	so.attribute("row_count", row_count);
	so.buffer("array", _array, row_count);
}
