//! \file PVAxisIndexType.h
//! $Id: PVAxesIndexType.h 3090 2011-06-09 04:59:46Z stricaud $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVCORE_PVAXESINDEXTYPE_H
#define PVCORE_PVAXESINDEXTYPE_H

#include <pvkernel/core/general.h>
#include <pvbase/types.h>
#include <pvkernel/core/PVArgument.h>

#include <vector>
#include <algorithm>

#include <QMetaType>
#include <QStringList>

namespace PVCore {

class PVAxesIndexType: public std::vector<PVCol>, public PVArgumentType<PVAxesIndexType>
{
public:
	PVAxesIndexType():
		std::vector<PVCol>()
	{ }

	// Used to create this type from the returned type of PVView::get_original_axes_index_with_tag
	PVAxesIndexType(QList<PVCol> const& cols)
	{
		reserve(cols.size());
		for (int i = 0; i < cols.size(); i++) {
			push_back(cols[i]);
		}
	}

	QString to_string() const
	{
		QStringList strList;
		for(int i = 0; i < this->size(); i++) {
			strList.append(QString::number((*this)[i]));
		}
		return strList.join(",");
	}
	PVArgument from_string(QString const& str) const
	{
		QStringList strList = str.split(",");
		PVAxesIndexType vec;
		for (int i = 0 ; i < strList.count(); i++) {
			vec.push_back(strList[i].toInt());
		}
		PVArgument arg;
		arg.setValue(PVAxesIndexType(vec));
		return arg;
	}
	bool operator==(const PVAxesIndexType &other) const
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
