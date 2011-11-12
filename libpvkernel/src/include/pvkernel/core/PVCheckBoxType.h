//! \file PVCheckBoxType.h
//! $Id$
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVCORE_PVCHECKBOXTYPE_H
#define PVCORE_PVCHECKBOXTYPE_H

#include <pvkernel/core/general.h>
#include <pvbase/types.h>
#include <QMetaType>

namespace PVCore {

/**
 * \class PVCheckBoxType
 * \note This class is fully implemented in its definition, so no LibKernelDecl is needed (each library will have its own version).
 */
class PVCheckBoxType
{
public:
	PVCheckBoxType() { _checked = false; };
	PVCheckBoxType(const bool checked) { set_checked(checked); };

	inline bool get_checked() const { return _checked; }
	inline void set_checked(const bool checked) { _checked = checked; }

protected:
	bool _checked;
};

}

// WARNING : This declaration MUST BE outside namespace's scope
Q_DECLARE_METATYPE(PVCore::PVCheckBoxType)

#endif // PVCORE_PVCHECKBOXTYPE_H
