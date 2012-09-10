#ifndef PVCORE_PVDECIMALSTORAGE_H
#define PVCORE_PVDECIMALSTORAGE_H

#include <boost/integer.hpp>

namespace PVCore {

namespace __impl {

template <size_t storage_bits>
class PVDecimalStorageBase
{
	static_assert((1<<(boost::static_log2<storage_bits>::value) == storage_bits) && (storage_bits <= 64), "PVCore::PVDecimalStorage: storage_bits must be a power of 2, and <= 64.");

	typedef typename boost::uint_t<storage_bits>::exact uint_type;
	typedef typename boost::int_t<storage_bits>::exact int_type;
	typedef uint_type storage_type;

public:
	uint_type& storage_as_uint() { return _v; }
	uint_type const& storage_as_uint() const { return _v; }

	int_type& storage_as_int() { return _v; }
	int_type const& storage_as_int() const { return _v; }

private:
	storage_type _v;
};

}

template <size_t storage_bits>
class PVDecimalStorage: public __impl::PVDecimaleStorageBase<storage_bits>
{
	typedef typename __impl::PVDecimalStorageBase<storage_bits>::storage_type storage_type;
	static_assert(sizeof(PVDecimalStorage<storage_bits>) == sizeof(storage_type));
};

template <>
class PVDecimalStorage<32>: public __impl::PVDecimaleStorageBase<32>
{
	typedef typename __impl::PVDecimalStorageBase<32>::storage_type storage_type;
	static_assert(sizeof(PVDecimalStorage<32>) == sizeof(storage_type));

public:
	float& storage_as_float() { return *(reinterpret_cast<float*>(&_v)); }
	float const& storage_as_float() const { return *(reinterpret_cast<float*>(&_v)); }
};

template <>
class PVDecimalStorage<64>: public __impl::PVDecimaleStorageBase<64>
{
	typedef typename __impl::PVDecimalStorageBase<64>::storage_type storage_type;
	static_assert(sizeof(PVDecimalStorage<64>) == sizeof(storage_type));

public:
	double& storage_as_double() { return *(reinterpret_cast<double*>(&_v)); }
	double const& storage_as_double() const { return *(reinterpret_cast<double*>(&_v)); }
};

}

#endif
