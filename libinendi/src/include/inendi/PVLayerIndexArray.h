/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef INENDI_PVLAYERINDEXARRAY_H
#define INENDI_PVLAYERINDEXARRAY_H

#include <inendi/general.h>

#include <pvkernel/core/PVSerializeArchive.h>

namespace Inendi {

/******************************************************************************
 *
 * WARNING!
 *
 * It is important to get that the value in layer_index_array have the
 *  following meaning :
 *   0 : SPECIAL VALUE! : means that the line is not present in any layer
 *                        in the layer stack
 *   1-256 : means that the line appears first (upmost/higher value) at
 *           that given value.
 *
 * So be careful about the indexing when using arrays...
 *
 *****************************************************************************/

/**
 * \class PVLayerIndexArray
 */
class PVLayerIndexArray
{
	friend class PVCore::PVSerializeObject;

public:
	int get_value(int row_index) const { return _array[row_index];}
	PVRow get_row_count() const { return _array.size(); }

	void initialize();

	void set_row_count(PVRow row_count);
	void set_value(PVRow row_index, int value);

protected:
	void serialize(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t v);

private:
	std::vector<int> _array;
};

} // namespace Inendi

#endif /* INENDI_PVLAYERINDEXARRAY_H_ */


