/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVCORE_PVAXISINDEXTYPE_H
#define PVCORE_PVAXISINDEXTYPE_H

#include <pvbase/types.h>

#include <pvkernel/core/PVArgument.h>

#include <QMetaType>
#include <QString>
#include <QStringList>

namespace PVCore
{

/**
 * \class PVAxisIndexType
 */
class PVAxisIndexType : public PVArgumentType<PVAxisIndexType>
{

  public:
	/**
	 * Constructor
	 */
	explicit PVAxisIndexType();
	explicit PVAxisIndexType(PVCol origin_axis_index,
	                         bool append_none_axis = false,
	                         PVCombCol axis_index = PVCombCol(0));

	PVCol get_original_index();
	PVCombCol get_axis_index();
	bool get_append_none_axis();

	QString to_string() const override
	{
		return QString::number(_origin_axis_index) + ":" +
		       QString(_append_none_axis ? "true" : "false");
	}
	PVArgument from_string(QString const& str, bool* ok /*= 0*/) const override
	{
		bool res_ok = false;

		PVArgument arg;

		QStringList parts = str.split(":");
		if (parts.count() == 2) {
			PVCol origin_axis_index(parts[0].toInt(&res_ok));
			bool append_none_axis = parts[1].compare("true", Qt::CaseInsensitive) == 0;
			arg.setValue(PVAxisIndexType(origin_axis_index, append_none_axis));
		}

		if (ok) {
			*ok = res_ok;
		}

		return arg;
	}
	bool operator==(const PVAxisIndexType& other) const
	{
		return _origin_axis_index == other._origin_axis_index &&
		       _append_none_axis == other._append_none_axis;
	}

  protected:
	// The original axis index will never change. PVAxisCombination takes care of any
	// axis addition/order modification, but will never change the original axis index.
	PVCol _origin_axis_index;
	PVCombCol _axis_index;
	bool _append_none_axis;
};
} // namespace PVCore

// WARNING : This declaration MUST BE outside namespace's scope
Q_DECLARE_METATYPE(PVCore::PVAxisIndexType)

#endif // PVCORE_PVAXISINDEXTYPE_H
