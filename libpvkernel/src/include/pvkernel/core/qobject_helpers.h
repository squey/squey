/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVCORE_QOBJECTSHELPER_H
#define PVCORE_QOBJECTSHELPER_H

#include <QObject>

namespace PVCore
{

/**
 * Search for a widget which can be cast to T* in the family tree of a widget.
 *
 * @param self the widget to test
 * @param test_self to indicate if \a self has to be checked or not
 */
template <typename T>
std::remove_pointer_t<T>* get_qobject_hierarchy_of_type(QObject* self, bool test_self = true)
{
	if (self == nullptr) {
		return nullptr;
	}

	for (QObject* o = test_self ? self : self->parent(); o != nullptr; o = o->parent()) {
		if (auto o_cast = dynamic_cast<std::remove_pointer_t<T>*>(o)) {
			return o_cast;
		}
	}
	return nullptr;
}

/**
 * Search for a widget which can be cast to T* in the ancestors of a widget.
 *
 * @param self the widget to test
 */
template <typename T>
std::remove_pointer_t<T>* get_qobject_parent_of_type(QObject* self)
{
	return get_qobject_hierarchy_of_type<T>(self, false);
}
} // namespace PVCore

#endif
