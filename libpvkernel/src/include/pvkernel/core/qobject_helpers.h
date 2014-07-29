#ifndef PVCORE_QOBJECTSHELPER_H
#define PVCORE_QOBJECTSHELPER_H

#include <QObject>

namespace PVCore {

/**
 * Search for a widget which can be cast to T* in the family tree of a widget.
 *
 * @param self the widget to test
 * @param test_self to indicate if \a self has to be checked or not
 */
template <typename T>
typename std::remove_pointer<T>::type* get_qobject_hierarchy_of_type(QObject* self,
                                                                     bool test_self = true)
{
	typedef typename std::remove_pointer<T>::type* pointer;

	if (self == nullptr) {
		return nullptr;
	}

	QObject* o = test_self?self:self->parent();
	while(o) {
		pointer o_cast = dynamic_cast<pointer>(o);
		if (o_cast) {
			return o_cast;
		}
		o = o->parent();
	}
	return nullptr;
}

/**
 * Search for a widget which can be cast to T* in the ancestors of a widget.
 *
 * @param self the widget to test
 */
template <typename T>
typename std::remove_pointer<T>::type* get_qobject_parent_of_type(QObject* self)
{
	return get_qobject_hierarchy_of_type<T>(self, false);
}

}

#endif
