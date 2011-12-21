#ifndef PVCOMPLIST_H
#define PVCOMPLIST_H

#include <pvkernel/core/PVArgument.h>
#include <QList>

namespace PVCore {

template <class T>
bool comp_list(T const& l1, T const& l2)
{
	typedef typename T::value_type Tv;
	typedef typename T::const_iterator Tit;

	if (l1.size() != l2.size()) {
		return false;
	}

	Tit it1,it2;
	it1 = l1.begin();
	it2 = l2.begin();

	for (; it1 != l1.end(); it1++) {
		Tv const& v1 = *it1;
		Tv const& v2 = *it2;
		if (v1 != v2) {
			return false;
		}
		it2++;
	}
	return true;
}

// AG: that's a hack so that format comparaison works. Waiting for better... :s
template <>
bool comp_list(QList<PVArgument> const& l1, QList<PVArgument> const& l2);

template <class K, class V>
bool comp_hash(QHash<K, V> const& h1, QHash<K, V> const& h2)
{
	if (!comp_list(h1.keys(), h2.keys())) {
		return false;
	}

	return comp_list(h1.values(), h2.values());
}

}

#endif
