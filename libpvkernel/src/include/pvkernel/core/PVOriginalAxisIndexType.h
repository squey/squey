/**
 * \file PVOriginalAxisIndexType.h
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#ifndef PVCORE_PVORIGINALAXISINDEXTYPE_H
#define PVCORE_PVORIGINALAXISINDEXTYPE_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVArgument.h>

#include <QMetaType>
#include <QStringList>

namespace PVCore {

/**
 * \class PVOriginalAxisIndexType
 */
class LibKernelDecl PVOriginalAxisIndexType : public PVArgumentType<PVOriginalAxisIndexType>
{
	
public:
	/**
	 * Constructor
	 */
	PVOriginalAxisIndexType(bool append_none_axis = false);
	PVOriginalAxisIndexType(int origin_axis_index, bool append_none_axis = false);

	int get_original_index() const;
	bool get_append_none_axis() const;

	QString to_string() const
	{
		return QString::number(_origin_axis_index) + ":" + QString(_append_none_axis?"true":"false");
	}
	PVArgument from_string(QString const& str, bool* ok /*= 0*/) const
	{
		bool res_ok = false;

		PVArgument arg;

		QStringList parts = str.split(":");
		if (parts.count() == 2) {
			int  origin_axis_index;
			bool append_none_axis;
			origin_axis_index = parts[0].toInt(&res_ok);
			append_none_axis = parts[1].compare("true", Qt::CaseInsensitive) == 0;
			arg.setValue(PVOriginalAxisIndexType(origin_axis_index, append_none_axis));
		}

		if (ok) {
			*ok = res_ok;
		}

		return arg;
	}

	bool operator==(const PVOriginalAxisIndexType &other) const
	{
		return _origin_axis_index == other._origin_axis_index && _append_none_axis == other._append_none_axis ;
	}

protected:
	int  _origin_axis_index;
	bool _append_none_axis;
};
}

// WARNING : This declaration MUST BE outside namespace's scope
Q_DECLARE_METATYPE(PVCore::PVOriginalAxisIndexType)


#endif // PVCORE_PVAXISINDEXTYPE_H
