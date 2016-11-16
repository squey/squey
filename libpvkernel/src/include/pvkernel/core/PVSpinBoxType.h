/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVCORE_PVSPINBOXTYPE_H
#define PVCORE_PVSPINBOXTYPE_H

#include <pvbase/types.h>
#include <QMetaType>

#include <pvkernel/core/PVArgument.h>

namespace PVCore
{

/**
 * \class PVSpinBoxType
 * \note This class is fully implemented in its definition, so no is needed (each library will have
 * its own version).
 */
class PVSpinBoxType : public PVArgumentType<PVSpinBoxType>
{
  public:
	PVSpinBoxType() : _value(0){};
	PVSpinBoxType(PVCol value) { set_value(value); }

	inline PVCol get_value() const { return _value; }
	inline void set_value(const PVCol value) { _value = value; }

	QString to_string() const override { return QString::number(_value); }
	PVArgument from_string(QString const& str, bool* ok /*= 0*/) const override
	{
		bool res_ok = false;

		PVArgument arg;
		arg.setValue(PVSpinBoxType(PVCol(str.toInt(&res_ok))));

		if (ok) {
			*ok = res_ok;
		}

		return arg;
	}
	bool operator==(const PVSpinBoxType& other) const { return _value == other._value; }

  protected:
	PVCol _value;
};
} // namespace PVCore

// WARNING : This declaration MUST BE outside namespace's scope
Q_DECLARE_METATYPE(PVCore::PVSpinBoxType)

#endif // PVCORE_PVSPINBOXTYPE_H
