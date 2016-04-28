/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVCORE_TEST_CONST_H
#define PVCORE_TEST_CONST_H

template <typename T>
struct defined_with_const {
	constexpr static bool value = false;
};

template <typename T>
struct defined_with_const<const T> {
	constexpr static bool value = true;
};

template <typename T>
struct defined_with_const<const T*> {
	constexpr static bool value = true;
};

template <typename T>
struct defined_with_const<const T&> {
	constexpr static bool value = true;
};

template <typename T>
bool is_const(T&)
{
	return defined_with_const<T>::value;
}

#endif // PVCORE_TEST_CONST_H
