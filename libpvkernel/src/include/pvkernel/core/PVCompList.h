/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVCOMPLIST_H
#define PVCOMPLIST_H

#include <pvkernel/core/PVArgument.h>
template <class Key, class T>
class QHash;

namespace PVCore
{

template <class K, class V>
bool comp_hash(QHash<K, V> const& h1, QHash<K, V> const& h2)
{
	if (h1.count() != h2.count()) {
		return false;
	}

	for (auto it1 = h1.constBegin(); it1 != h1.constEnd(); it1++) {
		auto it2 = h2.find(it1.key());
		if (it1.value() != it2.value()) {
			return false;
		}
	}

	return true;
}

bool comp_hash(PVCore::PVArgumentList const& h1, PVCore::PVArgumentList const& h2);

bool comp_hash(PVCore::PVArgumentList const& h1,
               PVCore::PVArgumentList const& h2,
               const PVCore::PVArgumentKeyList& keys);
} // namespace PVCore

#endif
