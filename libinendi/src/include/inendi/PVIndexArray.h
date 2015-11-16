/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef INENDI_PVINDEXARRAY_H
#define INENDI_PVINDEXARRAY_H

#include <QtCore>

#include <pvkernel/core/general.h>

#include <inendi/PVSelection.h>
#include <assert.h>

#define INENDI_INDEX_ARRAY_MAX_SIZE INENDI_LINES_MAX

namespace Inendi {

/**
 * \class PVIndexArray
 */
class PVIndexArray {
private:
	int array [INENDI_INDEX_ARRAY_MAX_SIZE];
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
	int at(int index) const { assert(index < INENDI_INDEX_ARRAY_MAX_SIZE); return array[index]; }

	void set_from_selection(const PVSelection & selection_);
	void set_row_count(int row_count_) { row_count = row_count_; }
};
}

#endif /* INENDI_PVINDEXARRAY_H_ */





// These are still used

// int inendi_index_array_get_index_count(inendi_index_array_t *ia);
//
// void inendi_index_array_set_from_selection(inendi_index_array_t *ia, inendi_selection_t *selection);
// 
