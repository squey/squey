/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVCORE_PVAXESINDEXTYPE_H
#define PVCORE_PVAXESINDEXTYPE_H

#include <pvkernel/core/general.h>
#include <pvbase/types.h>
#include <pvkernel/core/PVArgument.h>

#include <vector>
#include <algorithm>

#include <QMetaType>
#include <QStringList>

namespace PVCore
{

class PVAxesIndexType : public std::vector<PVCol>, public PVArgumentType<PVAxesIndexType>
{
  public:
	PVAxesIndexType() : std::vector<PVCol>() {}

	QString to_string() const
	{
		QStringList strList;
		for (size_t i = 0; i < this->size(); i++) {
			strList.append(QString::number((*this)[i]));
		}
		return strList.join(",");
	}
	PVArgument from_string(QString const& str, bool* ok /*= 0*/) const
	{
		bool res_ok = true;
		QStringList strList = str.split(",");
		PVAxesIndexType vec;
		bool tmp_ok = true;
		for (int i = 0; i < strList.count(); i++) {
			vec.push_back(strList[i].toInt(&tmp_ok));
			res_ok &= tmp_ok;
		}

		if (ok) {
			*ok = res_ok && strList.count() > 0;
		}

		PVArgument arg;
		arg.setValue(PVAxesIndexType(vec));
		return arg;
	}
	bool operator==(const PVAxesIndexType& other) const
	{
		if (this->size() != other.size()) {
			return false;
		}
		return std::equal(this->begin(), this->end(), other.begin());
	}
};
}

// WARNING : This declaration MUST BE outside namespace's scope
Q_DECLARE_METATYPE(PVCore::PVAxesIndexType)

#endif // PVCORE_PVAXESINDEXTYPE_H
