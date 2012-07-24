/**
 * \file PVCompList.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/core/PVCompList.h>
#include <pvkernel/core/PVTimeFormatType.h>

namespace PVCore {

template <>
bool comp_list(QList<PVArgument> const& l1, QList<PVArgument> const& l2)
{
	if (l1.size() != l2.size()) {
		return false;
	}

	QList<QVariant>::const_iterator it1,it2;
	it1 = l1.begin();
	it2 = l2.begin();

for (; it1 != l1.end(); it1++) {
		QVariant const& v1 = *it1;
		QVariant const& v2 = *it2;

		if (v1.userType() >= QMetaType::User && v1.userType() == v2.userType()) { // custom type
			const PVArgumentTypeBase* v1b = static_cast<const PVArgumentTypeBase*>(v1.constData());
			const PVArgumentTypeBase* v2b = static_cast<const PVArgumentTypeBase*>(v2.constData());
			if (!v1b->is_equal(*v2b)) {
				return false;
			}
		}
		else
		if (v1 != v2) {
			return false;
		}
		it2++;
	}

	return true;
}

template <>
bool comp_hash(PVCore::PVArgumentList const& h1, PVCore::PVArgumentList const& h2)
{
	if (h1.count() != h2.count()) {
		return false;
	}

	for (PVCore::PVArgumentList::const_iterator it1 = h1.constBegin(); it1 != h1.constEnd(); it1++) {
		PVCore::PVArgumentList::const_iterator it2 = h2.find(it1.key());
		if (it2 == h2.constEnd()) {
			return false;
		}
		if (it1.value().userType() >= QMetaType::User && it1.value().userType() == it2.value().userType()) { // custom type
			const PVArgumentTypeBase* v1 = static_cast<const PVArgumentTypeBase*>(it1.value().constData());
			const PVArgumentTypeBase* v2 = static_cast<const PVArgumentTypeBase*>(it2.value().constData());
			if (!v1->is_equal(*v2)) {
				return false;
			}
		}
		else if (it1.value() != it2.value()) {
			return false;
		}
	}

	return true;
}

}
