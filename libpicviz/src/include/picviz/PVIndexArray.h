//! \file PVIndexArray.h
//! $Id: PVIndexArray.h 3090 2011-06-09 04:59:46Z stricaud $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PICVIZ_PVINDEXARRAY_H
#define PICVIZ_PVINDEXARRAY_H

#include <QtCore>

#include <pvkernel/core/general.h>

#include <picviz/PVSelection.h>
#include <assert.h>

#define PICVIZ_INDEX_ARRAY_MAX_SIZE PICVIZ_LINES_MAX

namespace Picviz {

/**
 * \class PVIndexArray
 */
class LibPicvizDecl PVIndexArray {
private:
	int array [PICVIZ_INDEX_ARRAY_MAX_SIZE];
	int index_count;
	int row_count;
	

public:
	/**
	 * Constructor
	 */
	PVIndexArray();

	/**
	 * Constructor
	 */
	PVIndexArray(PVRow initial_row_count);

	int get_index_count() const {return index_count;}
	PVRow get_row_count() const {return row_count;}
	int at(int index) const { assert(index < PICVIZ_INDEX_ARRAY_MAX_SIZE); return array[index]; }

	void set_from_selection(const PVSelection & selection_);
	void set_row_count(int row_count_) { row_count = row_count_; }
};
}

#endif /* PICVIZ_PVINDEXARRAY_H_ */





// These are still used

// LibPicvizDecl int picviz_index_array_get_index_count(picviz_index_array_t *ia);
//
// LibPicvizDecl void picviz_index_array_set_from_selection(picviz_index_array_t *ia, picviz_selection_t *selection);
// 
