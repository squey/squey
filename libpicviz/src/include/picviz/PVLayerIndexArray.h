//! \file PVLayerIndexArray.h
//! $Id: PVLayerIndexArray.h 3090 2011-06-09 04:59:46Z stricaud $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PICVIZ_PVLAYERINDEXARRAY_H
#define PICVIZ_PVLAYERINDEXARRAY_H

#include <QtCore>

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
};
}

#endif /* PICVIZ_PVLAYERINDEXARRAY_H_ */


