//! \file PVAxisIndexCheckBoxType.h
//! $Id$
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVCORE_PVAXISINDEXCHECKBOXTYPE_H
#define PVCORE_PVAXISINDEXCHECKBOXTYPE_H

#include <pvcore/general.h>
#include <QMetaType>

namespace PVCore {

/**
 * \class PVAxisIndexCheckBoxType
 */
class LibCoreDecl PVAxisIndexCheckBoxType
{
	
public:
	/**
	 * Constructor
	 */
	PVAxisIndexCheckBoxType();
	PVAxisIndexCheckBoxType(int origin_axis_index, bool is_checked);

	inline int get_original_index() { return _origin_axis_index; }

	inline bool get_checked() { return _is_checked; }
	inline void set_checked(const bool checked) { _is_checked = checked; }

protected:
	// The original axis index will never change. PVAxisCombination takes care of any
	// axis addition/order modification, but will never change the original axis index.
	int  _origin_axis_index;
	bool _is_checked;
};
}

// WARNING : This declaration MUST BE outside namespace's scope
Q_DECLARE_METATYPE(PVCore::PVAxisIndexCheckBoxType)


#endif // PVCORE_PVAXISINDEXCHECKBOXTYPE_H
