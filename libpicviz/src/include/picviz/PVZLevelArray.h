//! \file PVZLevelArray.h
//! $Id: PVZLevelArray.h 3090 2011-06-09 04:59:46Z stricaud $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PICVIZ_PVZLEVELARRAY_H
#define PICVIZ_PVZLEVELARRAY_H

#include <QtCore>

#include <pvkernel/core/general.h>

#include <picviz/PVSelection.h>

#include <assert.h>

#define PICVIZ_Z_LEVEL_ARRAY_MAX_SIZE PICVIZ_LINES_MAX


namespace Picviz {

/**
 * \class PVZLevelArray
 */
class LibPicvizDecl PVZLevelArray {
private:
	float array [PICVIZ_Z_LEVEL_ARRAY_MAX_SIZE];
	PVRow row_count;

public:

	/**
	 * Constructor
	 */
	PVZLevelArray(PVRow initial_row_count);

	float& get_value(int index) {assert(index < PICVIZ_Z_LEVEL_ARRAY_MAX_SIZE); return array[index];}
	const float& get_value(int index) const {assert(index < PICVIZ_Z_LEVEL_ARRAY_MAX_SIZE); return array[index];}
	
	int get_row_count() const {return row_count;}

	void set_level(PVRow row_index, float level);
	void set_row_count(PVRow new_row_count) {row_count = new_row_count;}
	
};
}

#endif /* PICVIZ_PVZLEVELARRAY_H_ */

