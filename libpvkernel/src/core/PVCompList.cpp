/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/PVCompList.h>
#include <pvkernel/core/PVTimeFormatType.h>

namespace PVCore
{

bool comp_hash(PVCore::PVArgumentList const& h1, PVCore::PVArgumentList const& h2)
{
	if (h1.size() != h2.size()) {
		return false;
	}

	for (PVCore::PVArgumentList::const_iterator it1 = h1.begin(); it1 != h1.end(); it1++) {
		PVCore::PVArgumentList::const_iterator it2 = h2.find(it1->key());
		if (it2 == h2.end()) {
			return false;
		}
		if (it1->value().userType() >= QMetaType::User &&
		    it1->value().userType() == it2->value().userType()) { // custom type
			const PVArgumentTypeBase* v1 =
			    static_cast<const PVArgumentTypeBase*>(it1->value().constData());
			const PVArgumentTypeBase* v2 =
			    static_cast<const PVArgumentTypeBase*>(it2->value().constData());
			if (!v1->is_equal(*v2)) {
				return false;
			}
		} else if (it1->value() != it2->value()) {
			return false;
		}
	}

	return true;
}

bool comp_hash(PVCore::PVArgumentList const& h1,
               PVCore::PVArgumentList const& h2,
               const PVCore::PVArgumentKeyList& keys)
{
	for (PVCore::PVArgumentKey key : keys) {
		PVCore::PVArgument arg1 = h1.at(key);
		PVCore::PVArgument arg2 = h2.at(key);

		if (!arg1.isValid() || !arg2.isValid()) {
			return false;
		}

		if (arg1.userType() >= QMetaType::User &&
		    arg1.userType() == arg2.userType()) { // custom type
			const PVArgumentTypeBase* v1 = static_cast<const PVArgumentTypeBase*>(arg1.constData());
			const PVArgumentTypeBase* v2 = static_cast<const PVArgumentTypeBase*>(arg2.constData());
			if (!v1->is_equal(*v2)) {
				return false;
			}
		} else if (arg1 != arg2) {
			return false;
		}
	}

	return true;
}
}
