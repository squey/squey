//! \file PVListFloat2D.h
//! $Id: PVListFloat2D.h 3090 2011-06-09 04:59:46Z stricaud $
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011


#ifndef PVCORE_PVLISTFLOAT2D_H
#define PVCORE_PVLISTFLOAT2D_H

#include <pvcore/general.h>
#include <pvcore/types.h>

namespace PVCore {
class LibExport PVListFloat2D {
	public:
		PVListFloat2D();
		~PVListFloat2D();
		/**
		 * allocate an arry of float
		 *@param width of array
		 *@param height of array
		 */
		void reserve(PVCol width, PVRow height);

		void free();

		int count();

		/**
		 * return an array with all data
		 */
		float* getData();

		/**
		 * return a pointer to a row
		 */
		float* getRowData(PVRow i);

		/**
		 * return a value in table.
		 */
		float getValue(PVRow i, PVCol j);

		/**
		 * to set a value.
		 *@param value to set.
		 *@param i
		 *@param j
		 */
		void setValue(float value, PVRow i, PVCol j);

		PVCol getWidth();

		PVRow getHeight();

	private:
		// Private copy constructor !!
		PVListFloat2D(const PVListFloat2D& o)
		{
			width = o.width;
			height = o.height;
			data = o.data;
		}

	private:
			PVCol width;
			PVRow height;
			float *data;
};//class PVListFloat2D

}//namespace PVCore

#endif	/* PICVIZ_PVPLOTTED_CREATE_TABLE_CUDA_H */
