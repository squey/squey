// builtin.h

#ifndef __INIT_MAX__
#define __INIT_MAX__

#include <iostream>

template <class T>
class init_max
{
public:
	// these functions are required to be members:

	init_max(T new_value = std::numeric_limits<T>::max());
	init_max(const init_max<T> & other);
	~init_max(void);

	init_max<T> & operator  =(const init_max<T> & other);
	init_max<T> & operator +=(const init_max<T> & other);
	init_max<T> & operator -=(const init_max<T> & other);
	init_max<T> & operator *=(const init_max<T> & other);
	init_max<T> & operator /=(const init_max<T> & other);
	init_max<T> & operator %=(const init_max<T> & other);
	init_max<T> & operator<<=(const init_max<T> & other);
	init_max<T> & operator>>=(const init_max<T> & other);
	init_max<T> & operator &=(const init_max<T> & other);
	init_max<T> & operator |=(const init_max<T> & other);
	init_max<T> & operator ^=(const init_max<T> & other);

	init_max<T> & operator++(void); // prefix  ++. returns an lvalue.
	init_max<T>   operator++(int);  // postfix ++. returns an rvalue.
	init_max<T> & operator--(void); // prefix  --. returns an lvalue.
	init_max<T>   operator--(int);  // postfix --. returns an rvalue.

	operator T(void) const;

	const T * operator&(void) const;
	      T * operator&(void)      ;

private:
	T value;
};

// members:

template <class T>
inline
init_max<T>::init_max(T new_value)
{
	value = new_value;
}

template <class T>
inline
init_max<T>::init_max(const init_max<T> & other)
{
	value = other.value;
}

template <class T>
inline
init_max<T>::~init_max(void)
{
}

template <class T>
inline
init_max<T> & init_max<T>::operator=(const init_max<T> & other)
{
	value = other.value;

	return *this;
}

template <class T>
inline
init_max<T> & init_max<T>::operator+=(const init_max<T> & other)
{
	value += other.value;

	return *this;
}

template <class T>
inline
init_max<T> & init_max<T>::operator-=(const init_max<T> & other)
{
	value -= other.value;

	return *this;
}

template <class T>
inline
init_max<T> & init_max<T>::operator*=(const init_max<T> & other)
{
	value *= other.value;

	return *this;
}

template <class T>
inline
init_max<T> & init_max<T>::operator/=(const init_max<T> & other)
{
	value /= other.value;

	return *this;
}

template <class T>
inline
init_max<T> & init_max<T>::operator%=(const init_max<T> & other)
{
	value %= other.value;

	return *this;
}

template <class T>
inline
init_max<T> & init_max<T>::operator<<=(const init_max<T> & other)
{
	value <<= other.value;

	return *this;
}

template <class T>
inline
init_max<T> & init_max<T>::operator>>=(const init_max<T> & other)
{
	value >>= other.value;

	return *this;
}

template <class T>
inline
init_max<T> & init_max<T>::operator&=(const init_max<T> & other)
{
	value &= other.value;

	return *this;
}

template <class T>
inline
init_max<T> & init_max<T>::operator|=(const init_max<T> & other)
{
	value |= other.value;

	return *this;
}

template <class T>
inline
init_max<T> & init_max<T>::operator^=(const init_max<T> & other)
{
	value ^= other.value;

	return *this;
}

template <class T>
inline
init_max<T> & init_max<T>::operator++(void) // prefix ++. returns an lvalue.
{
	++value;

	return *this;
}

template <class T>
inline
init_max<T> init_max<T>::operator++(int) // postfix ++. returns an rvalue.
{
	// if your compiler doesn't support this syntax:
	T result(value);
	// try this one:
//	T result = value;

	value++;

	return result;
}

template <class T>
inline
init_max<T> & init_max<T>::operator--(void) // prefix --. returns an lvalue.
{
	--value;

	return *this;
}

template <class T>
inline
init_max<T> init_max<T>::operator--(int) // postfix --. returns an rvalue.
{
	// if your compiler doesn't support this syntax:
	T result(value);
	// try this one:
//	T result = value;

	value--;

	return result;
}

template <class T>
inline
init_max<T>::operator T(void) const
{
	return value;
}

template <class T>
inline
const T * init_max<T>::operator&(void) const
{
	return &value;
}

template <class T>
inline
T * init_max<T>::operator&(void)
{
	return &value;
}

// non-members:

// unary:

template <class T>
inline
init_max<T> operator+(const init_max<T> & bi)
{
	return bi;
}

template <class T>
inline
init_max<T> operator-(const init_max<T> & bi)
{
	return -((T) bi);
}

template <class T>
inline
init_max<T> operator~(const init_max<T> & bi)
{
	return ~((T) bi);
}

template <class T>
inline
init_max<T> operator!(const init_max<T> & bi)
{
	return !((T) bi);
}

// binary:

template <class T>
inline
std::istream & operator>>(std::istream & s, init_max<T> & bi)
{
	T local_t;
	s >> local_t;
	bi = local_t;

	return s;
}

template <class T>
inline
std::ostream & operator<<(std::ostream & s, const init_max<T> & bi)
{
	s << ((T) bi);

	return s;
}

template <class T>
inline
init_max<T> operator>>(const init_max<T> & bi, int i) // bit shift.
{
	return ((T) bi) >> i;
}

template <class T>
inline
init_max<T> operator<<(const init_max<T> & bi, int i) // bit shift.
{
	return ((T) bi) << i;
}

template <class T>
inline
bool operator==(const init_max<T> & bi1, const init_max<T> & bi2)
{
	return ((T) bi1) == ((T) bi2);
}

template <class T>
inline
bool operator!=(const init_max<T> & bi1, const init_max<T> & bi2)
{
	return ((T) bi1) != ((T) bi2);
}

template <class T>
inline
bool operator<(const init_max<T> & bi1, const init_max<T> & bi2)
{
	return ((T) bi1) < ((T) bi2);
}

template <class T>
inline
bool operator<=(const init_max<T> & bi1, const init_max<T> & bi2)
{
	return ((T) bi1) <= ((T) bi2);
}

template <class T>
inline
bool operator>(const init_max<T> & bi1, const init_max<T> & bi2)
{
	return ((T) bi1) > ((T) bi2);
}

template <class T>
inline
bool operator>=(const init_max<T> & bi1, const init_max<T> & bi2)
{
	return ((T) bi1) >= ((T) bi2);
}

template <class T>
inline
init_max<T> operator+(const init_max<T> & bi1, const init_max<T> & bi2)
{
	return ((T) bi1) + ((T) bi2);
}

template <class T>
inline
init_max<T> operator-(const init_max<T> & bi1, const init_max<T> & bi2)
{
	return ((T) bi1) - ((T) bi2);
}

template <class T>
inline
init_max<T> operator*(const init_max<T> & bi1, const init_max<T> & bi2)
{
	return ((T) bi1) * ((T) bi2);
}

template <class T>
inline
init_max<T> operator/(const init_max<T> & bi1, const init_max<T> & bi2)
{
	return ((T) bi1) / ((T) bi2);
}

template <class T>
inline
init_max<T> operator%(const init_max<T> & bi1, const init_max<T> & bi2)
{
	return ((T) bi1) % ((T) bi2);
}

template <class T>
inline
init_max<T> operator&(const init_max<T> & bi1, const init_max<T> & bi2)
{
	return ((T) bi1) & ((T) bi2);
}

template <class T>
inline
init_max<T> operator^(const init_max<T> & bi1, const init_max<T> & bi2)
{
	return ((T) bi1) ^ ((T) bi2);
}

template <class T>
inline
init_max<T> operator|(const init_max<T> & bi1, const init_max<T> & bi2)
{
	return ((T) bi1) | ((T) bi2);
}

#endif // __INIT_MAX__
