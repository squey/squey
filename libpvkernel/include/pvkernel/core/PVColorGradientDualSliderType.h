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

#ifndef PVCORE_PVCOLORGRADIENTDUALSLIDERTYPE_H
#define PVCORE_PVCOLORGRADIENTDUALSLIDERTYPE_H

#include <pvkernel/core/PVArgument.h>

namespace PVCore
{

/**
 * \class PVColorGradientDualSliderType
 */
class PVColorGradientDualSliderType : public PVArgumentType<PVColorGradientDualSliderType>
{
  public:
	PVColorGradientDualSliderType()
	{
		_sliders_positions[0] = 0;
		_sliders_positions[1] = 1;
	};
	explicit PVColorGradientDualSliderType(const double positions[2]) { set_positions(positions); }

	inline const double* get_positions() const { return _sliders_positions; }
	inline void set_positions(const double pos[2])
	{
		_sliders_positions[0] = pos[0];
		_sliders_positions[1] = pos[1];
	}

	QString to_string() const override
	{
		return QString::number(_sliders_positions[0]) + "," +
		       QString::number(_sliders_positions[1]);
	}
	PVArgument from_string(QString const& str, bool* ok /*= 0*/) const override
	{
		PVArgument arg;
		bool ok1 = false;
		bool ok2 = false;

		QStringList strList = str.split(",");
		if (strList.count() == 2) {
			double pos[2] = {strList[0].toDouble(&ok1), strList[1].toDouble(&ok2)};
			arg.setValue(PVColorGradientDualSliderType(pos));
		}

		if (ok) {
			*ok = ok1 && ok2;
		}

		return arg;
	}
	bool operator==(const PVColorGradientDualSliderType& other) const
	{
		return _sliders_positions[0] == other._sliders_positions[0] &&
		       _sliders_positions[1] == other._sliders_positions[1];
	}

  protected:
	double _sliders_positions[2];
};
} // namespace PVCore

// WARNING : This declaration MUST BE outside namespace's scope
Q_DECLARE_METATYPE(PVCore::PVColorGradientDualSliderType)

#endif // PVCORE_PVCOLORGRADIENTDUALSLIDERTYPE_H
