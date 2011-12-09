//! \file PVListFloat2D.h
//! $Id: PVListFloat2D.h 3090 2011-06-09 04:59:46Z stricaud $
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011


#ifndef PVCORE_PVLISTFLOAT2D_H
#define PVCORE_PVLISTFLOAT2D_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVMatrix.h>
#include <QVector>

namespace PVCore {

class PVListFloat2D: PVMatrix<float, PVRow, PVCol>
{
public:
	PVListFloat2D():
		PVMatrix<float, PVRow, PVCol>()
	{
	}

	PVListFloat2D(const PVListFloat2D& o):
		PVMatrix<float, PVRow, PVCol>()
	{
		o.copy_to(*this);
	}
	
	inline bool reserve(PVCol width, PVRow height)
	{
		return resize(height, width);
	}

	inline size_t count() const { return get_nrows()*get_ncols(); }

	/**
	 * return an array with all data
	 */
	inline float* getData() { return get_data(); };
	inline const float* getData() const { return get_data(); }

	/**
	 * return a pointer to a row
	 */
	inline float* getRowData(PVRow i) { return get_row_ptr(i); }
	inline const float* getRowData(PVRow i) const { return get_row_ptr(i); }

	/**
	 * return a value in table.
	 */
	inline float getValue(PVRow i, PVCol j) const { return at(i, j); }

	/**
	 * to set a value.
	 *@param value to set.
	 *@param i
	 *@param j
	 */
	inline void setValue(float value, PVRow i, PVCol j) { set_value(i, j, value); }

	inline PVCol getWidth() const { return get_ncols(); }

	inline PVRow getHeight() const { return get_nrows(); }
};

}

#endif	/* PICVIZ_PVPLOTTED_CREATE_TABLE_CUDA_H */
