/* * MIT License
 *
 * © ESI Group, 2015
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
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
