#ifndef PVCORE_PVDECIMALSTORAGE_H
#define PVCORE_PVDECIMALSTORAGE_H

#include <boost/integer.hpp>
#include <boost/integer/static_log2.hpp>

#include <limits>
#include <cassert>

// Disable strict-aliasing warning (GCC)
#ifdef __GNUG__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif

namespace PVCore {

enum DecimalType {
	IntegerType = 0,
	UnsignedIntegerType,
	FloatType
};

namespace __impl {

template <size_t storage_bits>
class PVDecimalStorageBase
{
	protected:
	static_assert((1<<(boost::static_log2<storage_bits>::value) == storage_bits) && (storage_bits <= 64), "PVCore::PVDecimalStorage: storage_bits must be a power of 2, and <= 64.");

	typedef typename boost::uint_t<storage_bits>::exact uint_type;
	typedef typename boost::int_t<storage_bits>::exact int_type;
	typedef uint_type storage_type;

public:
	uint_type& storage_as_uint() { return _v; }
	uint_type const& storage_as_uint() const { return _v; }

	int_type& storage_as_int() { return *(reinterpret_cast<int_type*>(&_v)); }
	int_type const& storage_as_int() const { return *(reinterpret_cast<int_type const*>(&_v)); }

	template <typename T, typename std::enable_if<std::is_same<T, int_type>::value, int>::type = 0>
	void set_max()
	{
		_v = std::numeric_limits<int_type>::max();
	}

	template <typename T, typename std::enable_if<std::is_same<T, uint_type>::value, int>::type = 0>
	void set_max()
	{
		_v = std::numeric_limits<uint_type>::max();
	}

	template <typename T, typename std::enable_if<std::is_same<T, int_type>::value, int>::type = 0>
	void set_min()
	{
		_v = std::numeric_limits<int_type>::min();
	}

	template <typename T, typename std::enable_if<std::is_same<T, uint_type>::value, int>::type = 0>
	void set_min()
	{
		_v = std::numeric_limits<uint_type>::min();
	}

	template <typename T>
	typename std::enable_if<std::is_same<T, uint_type>::value, T>::type& storage_cast() { return storage_as_uint(); };

	template <typename T>
	typename std::enable_if<std::is_same<T, int_type>::value, T>::type&  storage_cast() { return storage_as_int(); };

	template <typename T>
	typename std::enable_if<std::is_same<T, uint_type>::value, T>::type const& storage_cast() const { return storage_as_uint(); };

	template <typename T>
	typename std::enable_if<std::is_same<T, int_type>::value, T>::type const&  storage_cast() const { return storage_as_int(); };

public:
	template <typename C, typename... P>
	static auto call_from_type(DecimalType const type, P && ... params) -> decltype(C::template call<int_type>(params...))
	{
		switch (type) {
			case IntegerType:
				return C::template call<int_type>(std::forward<P>(params)...);
			case UnsignedIntegerType:
				return C::template call<uint_type>(std::forward<P>(params)...);
			default:
				assert(false);
		}

		return decltype(C::template call<int_type>(params...))();
	}

protected:
	storage_type _v;
};

}

template <size_t storage_bits>
class PVDecimalStorage: public __impl::PVDecimalStorageBase<storage_bits>
{
	typedef typename __impl::PVDecimalStorageBase<storage_bits>::storage_type storage_type;
	static_assert(sizeof(PVDecimalStorage<storage_bits>) == sizeof(storage_type), "PVDecimalStorage has an invalid size !");
};

template <>
class PVDecimalStorage<32>: public __impl::PVDecimalStorageBase<32>
{
	typedef __impl::PVDecimalStorageBase<32> base_type;
	typedef base_type::storage_type storage_type;

	static_assert(sizeof(float) == sizeof(storage_type), "float isn't stored on 32 bits !");

public:
	float& storage_as_float() { return *(reinterpret_cast<float*>(&_v)); }
	float const& storage_as_float() const { return *(reinterpret_cast<float const*>(&_v)); }

	template <typename T>
	typename std::enable_if<std::is_same<T, float>::value == true,  T>::type& storage_cast() { return storage_as_float(); };

	template <typename T>
	typename std::enable_if<std::is_same<T, float>::value == false, T>::type& storage_cast() { return base_type::template storage_cast<T>(); }

	template <typename T>
	typename std::enable_if<std::is_same<T, float>::value == true,  T>::type const& storage_cast() const { return storage_as_float(); };

	template <typename T>
	typename std::enable_if<std::is_same<T, float>::value == false, T>::type const& storage_cast() const { return base_type::template storage_cast<T>(); }

	template <typename T, typename std::enable_if<std::is_same<T, float>::value == true,  int>::type = 0>
	void set_max()
	{
		_v = std::numeric_limits<float>::max();
	}

	template <typename T, typename std::enable_if<std::is_same<T, float>::value == false, int>::type = 0>
	void set_max()
	{
		base_type::set_max<T>();
	}

	template <typename T, typename std::enable_if<std::is_same<T, float>::value == true,  int>::type = 0>
	void set_min()
	{
		_v = std::numeric_limits<float>::min();
	}

	template <typename T, typename std::enable_if<std::is_same<T, float>::value == false, int>::type = 0>
	void set_min()
	{
		base_type::set_min<T>();
	}

};

template <>
class PVDecimalStorage<64>: public __impl::PVDecimalStorageBase<64>
{
	typedef __impl::PVDecimalStorageBase<64> base_type;
	typedef typename base_type::storage_type storage_type;
	static_assert(sizeof(double) == sizeof(storage_type), "double isn't stored on 64 bits !");

public:
	double& storage_as_float() { return *(reinterpret_cast<double*>(&_v)); }
	double const& storage_as_float() const { return *(reinterpret_cast<double const*>(&_v)); }

	template <typename T>
	typename std::enable_if<std::is_same<T, double>::value == true,  T>::type& storage_cast() { return storage_as_float(); };

	template <typename T>
	typename std::enable_if<std::is_same<T, double>::value == false, T>::type& storage_cast() { return base_type::template storage_cast<T>(); }

	template <typename T>
	typename std::enable_if<std::is_same<T, double>::value == true,  T>::type const& storage_cast() const { return storage_as_float(); };

	template <typename T>
	typename std::enable_if<std::is_same<T, double>::value == false, T>::type const& storage_cast() const { return base_type::template storage_cast<T>(); }
};

}

#ifdef __GNUG__
#pragma GCC diagnostic pop
#endif

#endif
