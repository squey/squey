/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2016
 */

#ifndef __PVBASE_TYPED_BUILTIN_H__
#define __PVBASE_TYPED_BUILTIN_H__

#include <QMetaType>
#include <QtGlobal>

#include <typeinfo>
#include <limits>

namespace __impl
{

/**
 * Base class for strongly typed builtin types.
 *
 * It allows to express a difference of meaning behind a builtin
 * type in a way that can be checked at compile time.
 */
template <typename T>
class PVTypedBuiltin
{
  public:
	using value_type = int;
	static const value_type INVALID_VALUE;

  public:
	explicit PVTypedBuiltin(value_type v) : _value(v)
	{
		static bool registered = [&]() {
			qRegisterMetaType<PVTypedBuiltin<T>>(
			    (std::string("PVTypedBuiltin_") + typeid(T()).name()).c_str());
			return true;
		}();
		(void)registered;
	}
	explicit PVTypedBuiltin() : PVTypedBuiltin(INVALID_VALUE) {}

  public:
	operator value_type() const { return value(); }
	value_type value() const { return _value; }

  public:
	PVTypedBuiltin<T> operator+(const T& col) { return PVTypedBuiltin<T>(_value + col._value); }
	T operator+(value_type col) { return T(_value + col); }

	T& operator+=(const T& col)
	{
		_value += col._value;
		return (T&)*this;
	}
	T& operator+=(value_type c)
	{
		_value += c;
		return (T) * this;
	}

	T operator++(int)
	{
		T tmp(*this);
		operator++();
		return tmp;
	}
	T& operator++()
	{
		_value++;
		return (T&)*this;
	}

	T operator-(const T& col) { return T(_value - col._value); }
	T operator-(value_type col) { return T(_value - col); }

	T& operator-=(const T& col)
	{
		_value -= col._value;
		return (T&)*this;
	}
	T& operator-=(value_type c)
	{
		_value -= c;
		return (T&)*this;
	}

	T operator--(int)
	{
		T tmp(*this);
		operator--();
		return tmp;
	}
	T& operator--()
	{
		_value--;
		return (T&)*this;
	}

	bool operator<(const T& c) const { return _value < c._value; }
	bool operator<(value_type c) const { return _value < c; }

	bool operator<=(const T& c) const { return _value <= c._value; }
	bool operator<=(value_type c) const { return _value <= c; }

	bool operator>(const T& c) const { return _value > c._value; }
	bool operator>(value_type c) const { return _value > c; }

	bool operator>=(const T& c) const { return _value >= c._value; }
	bool operator>=(value_type c) const { return _value >= c; }

	bool operator==(const T& c) const { return _value == c._value; }
	bool operator==(value_type c) const { return c == _value; }

	bool operator!=(const T& c) const { return _value != c._value; }
	bool operator!=(value_type c) const { return c != _value; }

  private:
	value_type _value;
};

template <typename T>
const typename PVTypedBuiltin<T>::value_type
    PVTypedBuiltin<T>::INVALID_VALUE = std::numeric_limits<PVTypedBuiltin<T>::value_type>::max();

} // namespace __impl

#endif // __PVBASE_TYPED_BUILTIN_H__
