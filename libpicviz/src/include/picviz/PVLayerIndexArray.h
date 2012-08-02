/**
 * \file PVLayerIndexArray.h
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#ifndef PICVIZ_PVLAYERINDEXARRAY_H
#define PICVIZ_PVLAYERINDEXARRAY_H

#include <QtCore>

#include <pvkernel/core/PVSerializeArchive.h>
#include <picviz/general.h>

#define PICVIZ_LAYER_INDEX_ARRAY_MAX_SIZE PICVIZ_LINES_MAX
/* WARNING! : should be the same as PICVIZ_LAYER_STACK_MAX_DEPTH
        but not used to avoid circular dependencies
*/
#define PICVIZ_LAYER_INDEX_ARRAY_MAX_VALUE 256



namespace Picviz {


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
class LibPicvizDecl PVLayerIndexArray {
	friend class PVCore::PVSerializeObject;
private:
	int array [PICVIZ_LAYER_INDEX_ARRAY_MAX_SIZE];
	int row_count;

public:

	/**
	 * Constructor
	 */
	PVLayerIndexArray(int initial_row_count);

	int get_value(int row_index) const { return array[row_index];}
	int get_row_count() const {return row_count;}

	void initialize();

	void set_row_count(int new_row_count);
	void set_value(int row_index, int value);

protected:
	void serialize(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t v);
};
}

#endif /* PICVIZ_PVLAYERINDEXARRAY_H_ */


