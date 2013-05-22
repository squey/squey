/**
 * \file PVZoneIndexType.h
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#ifndef PVCORE_PVZONEINDEXTYPE_H
#define PVCORE_PVZONEINDEXTYPE_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVArgument.h>

#include <QMetaType>
#include <QStringList>

namespace PVCore {

/**
 * \class PVZoneIndexType
 */
class LibKernelDecl PVZoneIndexType : public PVArgumentType<PVZoneIndexType>
{
public:
	/**
	 * Constructor
	 */
	PVZoneIndexType(int zone_index = 0):
		_zone_index(zone_index)
	{ }

	int get_zone_index() const { return _zone_index; }

	QString to_string() const
	{
		return QString::number(get_zone_index());
	}

	PVArgument from_string(QString const& str, bool* ok /*= 0*/) const
	{
		PVArgument arg;
		bool res_ok;

		int zone_index = str.toInt(&res_ok);

		if (zone_index < 0) {
			res_ok = false;
			zone_index = 0;
		}

		arg.setValue(PVZoneIndexType(zone_index));

		if (ok) {
			*ok = res_ok;
		}

		return arg;
	}
	
	bool operator==(const PVZoneIndexType &other) const
	{
		return _zone_index == other._zone_index;
	}

protected:
	int _zone_index;
};

}

// WARNING : This declaration MUST BE outside namespace's scope
Q_DECLARE_METATYPE(PVCore::PVZoneIndexType)


#endif // PVCORE_PVZONEINDEXTYPE_H
