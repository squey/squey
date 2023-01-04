/* * MIT License
 *
 * © ESI Group, 2015
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
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
