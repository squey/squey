#ifndef PVCOMPLIST_H
#define PVCOMPLIST_H

#include <pvkernel/core/PVArgument.h>
#include <QList>

namespace PVCore {

/*! \brief Compares two C++ standard compliant containers.
 *  \tparam T A C++ standard compliant container. T::value_type::operator!= must exists.
 *  \param[in] l1 The first container
 *  \param[in] l2 The second container
 *  \return true if both containers have the same size and that all their elements are equal (and in the same order), false otherwise.
 */
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
	typedef typename QHash<K, V>::const_iterator Tit;

	if (h1.count() != h2.count()) {
		return false;
	}

	for (Tit it1 = h1.constBegin(); it1 != h1.constEnd(); it1++) {
		Tit it2 = h2.find(it1.key());
		if (it1.value() != it2.value()) {
			return false;
		}
	}

	return true;
}

template <>
bool comp_hash(PVCore::PVArgumentList const& h1, PVCore::PVArgumentList const& h2);

}

#endif
