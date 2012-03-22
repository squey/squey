//! \file PVAxisIndexType.h
//! $Id: PVAxisIndexType.h 3090 2011-06-09 04:59:46Z stricaud $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVCORE_PVAXISINDEXTYPE_H
#define PVCORE_PVAXISINDEXTYPE_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVArgument.h>

#include <QMetaType>
#include <QStringList>

namespace PVCore {

/**
 * \class PVAxisIndexType
 */
class LibKernelDecl PVAxisIndexType : public PVArgumentType<PVAxisIndexType>
{
	
public:
	/**
	 * Constructor
	 */
	PVAxisIndexType(bool append_none_axis = false);
	PVAxisIndexType(int origin_axis_index, bool append_none_axis = false);

	int get_original_index();
	bool get_append_none_axis();

	QString to_string() const
	{
		return QString::number(_origin_axis_index) + ":" + QString(_append_none_axis?"true":"false");
	}
	PVArgument from_string(QString const& str) const
	{
		int  origin_axis_index ;
		bool append_none_axis ;

		QStringList parts = str.split(":");
		origin_axis_index = parts[0].toInt();
		append_none_axis = parts[1].compare("true", Qt::CaseInsensitive) == 0;

		PVArgument arg;
		arg.setValue(PVAxisIndexType(origin_axis_index, append_none_axis));
		return arg;
	}
	bool operator==(const PVAxisIndexType &other) const
	{
		return _origin_axis_index == other._origin_axis_index && _append_none_axis == other._append_none_axis ;
	}

protected:
	// The original axis index will never change. PVAxisCombination takes care of any
	// axis addition/order modification, but will never change the original axis index.
	int  _origin_axis_index;
	bool _append_none_axis;
};
}

// WARNING : This declaration MUST BE outside namespace's scope
Q_DECLARE_METATYPE(PVCore::PVAxisIndexType)


#endif // PVCORE_PVAXISINDEXTYPE_H
