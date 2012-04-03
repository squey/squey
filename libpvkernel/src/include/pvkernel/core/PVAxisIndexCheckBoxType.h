//! \file PVAxisIndexCheckBoxType.h
//! $Id$
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVCORE_PVAXISINDEXCHECKBOXTYPE_H
#define PVCORE_PVAXISINDEXCHECKBOXTYPE_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVArgument.h>

#include <QMetaType>
#include <QStringList>

namespace PVCore {

/**
 * \class PVAxisIndexCheckBoxType
 */
class LibKernelDecl PVAxisIndexCheckBoxType : public PVArgumentType<PVAxisIndexCheckBoxType>
{
	
public:
	/**
	 * Constructor
	 */
	PVAxisIndexCheckBoxType();
	PVAxisIndexCheckBoxType(int origin_axis_index, bool is_checked);

	inline int get_original_index() { return _origin_axis_index; }

	inline bool get_checked() { return _is_checked; }
	inline void set_checked(const bool checked) { _is_checked = checked; }

	QString to_string() const
	{
		return QString::number(_origin_axis_index) + ":" + QString(_is_checked?"true":"false");
	}
	PVArgument from_string(QString const& str, bool* ok /*= 0*/) const
	{
		PVArgument arg;
		bool res_ok = false;

		QStringList parts = str.split(":");
		if (parts.count() == 2) {
			int  origin_axis_index ;
			bool is_checked ;
			origin_axis_index = parts[0].toInt(&res_ok);
			is_checked = parts[1].compare("true", Qt::CaseInsensitive) == 0;
			arg.setValue(PVAxisIndexCheckBoxType(origin_axis_index, is_checked));
		}

		if (ok) {
			*ok = res_ok;
		}

		return arg;
	}
	bool operator==(const PVAxisIndexCheckBoxType &other) const
	{
		return _origin_axis_index == other._origin_axis_index && _is_checked == other._is_checked ;
	}

protected:
	// The original axis index will never change. PVAxisCombination takes care of any
	// axis addition/order modification, but will never change the original axis index.
	int  _origin_axis_index;
	bool _is_checked;
};
}

// WARNING : This declaration MUST BE outside namespace's scope
Q_DECLARE_METATYPE(PVCore::PVAxisIndexCheckBoxType)


#endif // PVCORE_PVAXISINDEXCHECKBOXTYPE_H
