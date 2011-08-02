//! \file PVSpinBoxType.h
//! $Id: PVSpinBoxType.h 3199 2011-06-24 07:14:57Z aguinet $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVCORE_PVSPINBOXTYPE_H
#define PVCORE_PVSPINBOXTYPE_H

#include <pvcore/general.h>
#include <pvbase/types.h>
#include <QMetaType>

namespace PVCore {

/**
 * \class PVSpinBoxType
 * \note This class is fully implemented in its definition, so no LibCoreDecl is needed (each library will have its own version).
 */
class PVSpinBoxType
{
public:
	PVSpinBoxType() { _value = 0; };
	PVSpinBoxType(const PVCol value) { set_value(value); }

	inline PVCol get_value() const { return _value; }
	inline void set_value(const PVCol value) { _value = value; }

protected:
	PVCol _value;
};

}

// WARNING : This declaration MUST BE outside namespace's scope
Q_DECLARE_METATYPE(PVCore::PVSpinBoxType)

#endif // PVCORE_PVSPINBOXTYPE_H
