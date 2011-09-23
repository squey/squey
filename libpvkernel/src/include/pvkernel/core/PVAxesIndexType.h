//! \file PVAxisIndexType.h
//! $Id: PVAxesIndexType.h 3090 2011-06-09 04:59:46Z stricaud $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVCORE_PVAXESINDEXTYPE_H
#define PVCORE_PVAXESINDEXTYPE_H

#include <pvkernel/core/general.h>
#include <pvbase/types.h>

#include <vector>

#include <QMetaType>

namespace PVCore {

class PVAxesIndexType: public std::vector<PVCol>
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
};

}

// WARNING : This declaration MUST BE outside namespace's scope
Q_DECLARE_METATYPE(PVCore::PVAxesIndexType)


#endif // PVCORE_PVAXESINDEXTYPE_H
