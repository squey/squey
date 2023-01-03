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

#ifndef PVCORE_PVZONEINDEXTYPE_H
#define PVCORE_PVZONEINDEXTYPE_H

#include <pvbase/types.h>
#include <pvkernel/core/PVArgument.h>

#include <QMetaType>
#include <QStringList>

namespace PVCore
{

/**
 * \class PVZoneIndexType
 */
class PVZoneIndexType : public PVArgumentType<PVZoneIndexType>
{
  public:
	/**
	 * Constructor
	 */
	explicit PVZoneIndexType(int zone_index_first = 0, int zone_index_second = 0)
	    : _zone_index_first(zone_index_first), _zone_index_second(zone_index_second)
	{
	}

	PVCol get_zone_index_first() const { return _zone_index_first; }
	PVCol get_zone_index_second() const { return _zone_index_second; }

	QString to_string() const override
	{
		return QString::number(_zone_index_first) + ":" + QString::number(_zone_index_second);
	}

	PVArgument from_string(QString const& str, bool* ok /*= 0*/) const override
	{
		PVArgument arg;
		bool res_ok;
		int zone_index_first = 0;
		int zone_index_second = 0;

		QStringList strList = str.split(':');
		if (strList.size() != 2) {
			res_ok = false;
			zone_index_first = 0;
			zone_index_second = 0;
		} else {
			zone_index_first = strList[0].toInt(&res_ok);
			zone_index_second = strList[1].toInt(&res_ok);
			if (zone_index_first < 0) {
				res_ok = false;
				zone_index_first = 0;
			}
			if (zone_index_second < 0) {
				res_ok = false;
				zone_index_second = 0;
			}
		}

		arg.setValue(PVZoneIndexType(zone_index_first, zone_index_second));

		if (ok) {
			*ok = res_ok;
		}

		return arg;
	}

	bool operator==(const PVZoneIndexType& other) const
	{
		return _zone_index_first == other._zone_index_first &&
		       _zone_index_second == other._zone_index_second;
	}

  protected:
	PVCol _zone_index_first;
	PVCol _zone_index_second;
};
} // namespace PVCore

// WARNING : This declaration MUST BE outside namespace's scope
Q_DECLARE_METATYPE(PVCore::PVZoneIndexType)

#endif // PVCORE_PVZONEINDEXTYPE_H
