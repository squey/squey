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

#ifndef PVCORE_PVPERCENTRANGETYPE_H
#define PVCORE_PVPERCENTRANGETYPE_H

#include <QStringList>

#include <pvkernel/core/PVArgument.h>

#include <array>

namespace PVCore
{

class PVPercentRangeType : public PVArgumentType<PVPercentRangeType>
{
  public:
	PVPercentRangeType()
	{
		_values[0] = 0.;
		_values[1] = 100.;
	}

	PVPercentRangeType(double min, double max) : _values{{min, max}} {}

	explicit PVPercentRangeType(const double values[2]) { set_values(values); }

	inline const double* get_values() const { return _values.data(); }

	inline void set_values(const double values[2])
	{
		_values[0] = values[0];
		_values[1] = values[1];
	}

	QString to_string() const override
	{
		return QString::number(_values[0]) + "," + QString::number(_values[1]);
	}

	PVArgument from_string(QString const& str, bool* ok) const override
	{
		PVArgument arg;
		bool ok1 = false;
		bool ok2 = false;

		QStringList strList = str.split(",");
		if (strList.count() == 2) {
			double pos[2] = {strList[0].toDouble(&ok1), strList[1].toDouble(&ok2)};

			arg.setValue(PVPercentRangeType(pos));
		}

		if (ok) {
			*ok = ok1 && ok2;
		}

		return arg;
	}
	bool operator==(const PVPercentRangeType& other) const
	{
		return (_values[0] == other._values[0]) && (_values[1] == other._values[1]);
	}

  private:
	std::array<double, 2> _values;
};
} // namespace PVCore

Q_DECLARE_METATYPE(PVCore::PVPercentRangeType)

#endif // PVCORE_PVPERCENTRANGETYPE_H
