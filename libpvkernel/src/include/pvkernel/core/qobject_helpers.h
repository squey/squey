#ifndef PVCORE_QOBJECTSHELPER_H
#define PVCORE_QOBJECTSHELPER_H

#include <QObject>

namespace PVCore {

template <typename T>
typename std::remove_pointer<T>::type* get_qobject_parent_of_type(QObject* self)
{
	typedef typename std::remove_pointer<T>::type* pointer;
	QObject* parent = self->parent();
	pointer parent_cast = dynamic_cast<pointer>(parent);
	if (parent_cast) {
		return parent_cast;
	}   
	return get_qobject_parent_of_type<T>(parent);
}

}

#endif
