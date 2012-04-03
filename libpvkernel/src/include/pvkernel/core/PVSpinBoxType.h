//! \file PVSpinBoxType.h
//! $Id: PVSpinBoxType.h 3199 2011-06-24 07:14:57Z aguinet $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVCORE_PVSPINBOXTYPE_H
#define PVCORE_PVSPINBOXTYPE_H

#include <pvkernel/core/general.h>
#include <pvbase/types.h>
#include <QMetaType>

#include <pvkernel/core/PVArgument.h>

namespace PVCore {

/**
 * \class PVSpinBoxType
 * \note This class is fully implemented in its definition, so no LibKernelDecl is needed (each library will have its own version).
 */
class PVSpinBoxType : public PVArgumentType<PVSpinBoxType>
{
public:
	PVSpinBoxType() { _value = 0; };
	PVSpinBoxType(const PVCol value) { set_value(value); }

	inline PVCol get_value() const { return _value; }
	inline void set_value(const PVCol value) { _value = value; }

	QString to_string() const
	{
		return QString::number(_value);
	}
	PVArgument from_string(QString const& str, bool* ok /*= 0*/) const
	{
		bool res_ok = false;

		PVArgument arg;
		arg.setValue(PVSpinBoxType(str.toInt(&res_ok)));

		if (ok) {
			*ok = res_ok;
		}

		return arg;
	}
	bool operator==(const PVSpinBoxType &other) const
	{
	    return _value == other._value;
	}

protected:
	PVCol _value;
};

}

// WARNING : This declaration MUST BE outside namespace's scope
Q_DECLARE_METATYPE(PVCore::PVSpinBoxType)

#endif // PVCORE_PVSPINBOXTYPE_H
