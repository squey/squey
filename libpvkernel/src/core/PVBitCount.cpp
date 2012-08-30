#include <pvkernel/core/picviz_intrin.h>
#include <pvkernel/core/PVBitCount.h>
#include <pvkernel/core/PVHardwareConcurrency.h>

#include <cassert>

size_t PVCore::PVBitCount::bit_count(size_t n, const uint64_t* data)
{
	size_t ret = 0;
// TODO: need to benchmark for a good grainsize !
//#pragma omp parallel for reduction(+:ret) num_threads(PVCore::PVHardwareConcurrency::get_physical_core_number())
	for (size_t i = 0; i < n; i++) {
		uint64_t v = data[i];
		ret += bit_count(v);
	}
	return ret;
}

// a and b are positions in bits and are inclusive (which means that b-a+1 bits are checked)
// No boundary checks are done, so be carefull !!
size_t PVCore::PVBitCount::bit_count_between(size_t a, size_t b, const uint64_t* data)
{
	assert(b >= a);
	size_t a_byte = a >> 6;
	size_t b_byte = b >> 6;

	constexpr static size_t tmp = (1 << 6) - 1; // Used for modulus operations (%64)

	if (a_byte == b_byte) {
		const uint64_t v0 = data[a_byte];
		const size_t shift0 = (a    & tmp);
		const size_t shift1 = ((~b) & tmp);
		const uint64_t va = (v0 >> shift0) << shift0;
		const uint64_t vb = (v0 << shift1) >> shift1;
		return PVCore::PVBitCount::bit_count(va & vb);
	}

	// Hard part is done here
	size_t ret = bit_count(b_byte - a_byte - 1, data + a_byte + 1);

	// Finish it
	const uint64_t v0 = data[a_byte] >> (a & tmp);
	const uint64_t v1 = data[b_byte] << ((~b) & tmp);
	ret += PVCore::PVBitCount::bit_count(v0);
	ret += PVCore::PVBitCount::bit_count(v1);

	return ret;
}
