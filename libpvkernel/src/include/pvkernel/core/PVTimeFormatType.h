/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVCORE_PVTIMEFORMATTYPE_H
#define PVCORE_PVTIMEFORMATTYPE_H

#include <QStringList>

#include <pvkernel/core/PVArgument.h>

namespace PVCore
{

struct PVTimeFormatType : public QStringList, public PVArgumentType<PVTimeFormatType> {
	PVTimeFormatType() : QStringList() {}
	PVTimeFormatType(QStringList const& list) : QStringList(list) {}

	QString to_string() const { return join("\n"); }

	PVArgument from_string(QString const& str, bool* ok /*= 0*/) const
	{
		PVArgument arg;
		arg.setValue(PVTimeFormatType(str.split('\n')));

		if (ok) {
			*ok = true;
		}

		return arg;
	}
};
}
// WARNING : This declaration MUST BE outside namespace's scope
Q_DECLARE_METATYPE(PVCore::PVTimeFormatType)

#endif
