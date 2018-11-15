/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
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
