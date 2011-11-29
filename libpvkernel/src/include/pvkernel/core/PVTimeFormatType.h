#ifndef PVCORE_PVTIMEFORMATTYPE_H
#define PVCORE_PVTIMEFORMATTYPE_H

#include <QMetaType>
#include <QStringList>

namespace PVCore {

struct PVTimeFormatType: public QStringList
{
	PVTimeFormatType(): QStringList() { }
	PVTimeFormatType(QStringList const& list):
		QStringList(list)
	{ }
};

}
// WARNING : This declaration MUST BE outside namespace's scope
Q_DECLARE_METATYPE(PVCore::PVTimeFormatType)

#endif
