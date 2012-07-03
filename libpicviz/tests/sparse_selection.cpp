#include <pvkernel/core/general.h>
#include <pvkernel/core/picviz_bench.h>
#include <pvkernel/core/picviz_assert.h>

#include <picviz/PVSparseSelection.h>
#include <picviz/PVSelection.h>

#include <tbb/parallel_sort.h>

#include <iostream>
#include <algorithm>

#include <cstdlib>
#include <ctime>

template <class L>
bool check_equals(const size_t n, const size_t* bits, L const& l)
{
	bool ret = (n == l.size());
	if (ret) {
		ret = std::equal(l.begin(), l.end(), bits);
	}
	if (!ret) {
		std::cerr << "Error in selection: " << std::endl;
		std::cerr << "Ref is: ";
		for (size_t i = 0; i < n; i++) {
			std::cerr << bits[i] << " ";
		}
		std::cerr << std::endl << "Selection is: ";
		for (const size_t b: l) {
			std::cerr << b << " ";
		}
		std::cerr << std::endl;
	}
	return ret;
}

void dump_sel(const char* str, Picviz::PVSparseSelection const& s)
{
	std::vector<size_t> bits;
	s.to_list(bits);
	std::cerr << str << ": ";
	for (size_t b: bits) {
		std::cerr << b << " ";
	}
	std::cerr << std::endl;
}

int main()
{
	std::cout << "sizeof map_chunks_t::value_t: " << sizeof(Picviz::PVSparseSelection::map_chunks_t::value_type) << std::endl;
	Picviz::PVSparseSelection s;

	// Set bits i ndifferent chunks and check that the returned list is okay.
	const size_t bits[] = {1,10,64,66,102305064UL, 10329495499UL};
	size_t nbits = sizeof(bits)/sizeof(size_t);

	for (size_t i = 0; i < nbits; i++) {
		s.set(bits[i]);
	}

	std::vector<size_t> set_bits; set_bits.reserve(nbits);
	s.to_list(set_bits);
	ASSERT_VALID(check_equals(nbits, &bits[0], set_bits));

	// Set randomely bits and check
	nbits = 10000;
	s.clear();
	srand(time(NULL));
	std::vector<size_t> bits_ref; bits_ref.reserve(nbits);
	for (size_t i = 0; i < nbits; i++) {
		size_t v = rand();
		bits_ref.push_back(v);
		s.set(v);
	}
	std::sort(bits_ref.begin(), bits_ref.end());
	set_bits.clear(); set_bits.reserve(nbits);
	s.to_list(set_bits);
	ASSERT_VALID(check_equals(bits_ref.size(), &bits_ref[0], set_bits));

	// Check the AND operator
	Picviz::PVSparseSelection s2;
	s.clear();

	s.set(1); s2.set(1);
	s.set(65); s2.set(65);
	s.set(66); s2.set(64);
	s.set(10000ULL); s2.set(10000000ULL);
	s.set(1245678901ULL); s2.set(1245678901ULL);
	s.set(1245678902ULL); s2.set(1245678902ULL);
	dump_sel("s", s);
	dump_sel("s2", s2);
	s &= s2;
	dump_sel("s &= s2", s);

	set_bits.clear();
	s.to_list(set_bits);
	{
		const size_t bits_ref[] = {1,65,1245678901ULL,1245678902ULL};
		ASSERT_VALID(check_equals(4, (size_t*) bits_ref, set_bits));
	}

	// Check the OR operator
	s.clear(); s2.clear();
	s.set(1);
	s.set(65); s2.set(67);
	s.set(66); s2.set(64);
	s.set(10000ULL);
	                 s2.set(10000000ULL);
	dump_sel("s", s);
	dump_sel("s2", s2);
	s |= s2;
	dump_sel("s |= s2", s);

	set_bits.clear();
	s.to_list(set_bits);
	{
		const size_t bits_ref[] = {1,64,65,66,67,10000ULL,10000000ULL};
		ASSERT_VALID(check_equals(sizeof(bits_ref)/sizeof(size_t), (size_t*) bits_ref, set_bits));
	}

	// Performance tests
	//
	
	// Random number generation that will be used for the following tests
	std::vector<size_t> random_bits;
	std::vector<size_t> random_sorted_bits(random_bits);
	nbits = 1000000;
	random_bits.reserve(nbits);
	random_sorted_bits.reserve(nbits);
	for (size_t i = 0; i < nbits; i++) {
		const size_t rn = rand();
		random_bits.push_back(rn);
		random_sorted_bits.push_back(rn);
	}
	tbb::parallel_sort(random_sorted_bits.begin(), random_sorted_bits.end());

	// Creation perforamnce test
	s.clear();
	BENCH_START(creation);
	for (size_t i = 0; i < nbits; i++) {
		s.set(random_bits[i]);
	}
	BENCH_END(creation, "creation-random", sizeof(size_t), nbits, sizeof(Picviz::PVSparseSelection::map_chunks_t::value_type), s.get_chunks().size());

	s.clear();
	BENCH_START(creation2);
	for (size_t i = 0; i < nbits; i++) {
		s.set(random_sorted_bits[i]);
	}
	BENCH_END(creation2, "creation-sorted", sizeof(size_t), nbits, sizeof(Picviz::PVSparseSelection::map_chunks_t::value_type), s.get_chunks().size());

	// Fast creation/removing with small number of objects
	s.clear();

	BENCH_START(fast_cr);
	for (size_t i = 0; i < nbits; i++) {
		const size_t r = random_bits[i];
		s.set(r);
		s.clear();
	}
	BENCH_END_TRANSFORM(fast_cr, "fast-creation", sizeof(size_t), nbits);

	BENCH_START(fast_cr2);
	for (size_t i = 0; i < nbits; i++) {
		const size_t r = random_sorted_bits[i];
		s.set(r);
		s.clear();
	}
	BENCH_END_TRANSFORM(fast_cr2, "fast-creation-sorted", sizeof(size_t), nbits);

	// Fast creation/removing with AND-ing the intermediare results
	s.clear();
	s2.clear();

	BENCH_START(fast_cr_anding);
	for (size_t i = 0; i < nbits; i++) {
		const size_t r = random_bits[i];
		s2.set(r);
		s2.set(r+1);
		s2.set(r+2);
		s2.set(r+4);
		s &= s2;
		s2.clear();
	}
	BENCH_END_TRANSFORM(fast_cr_anding, "fast-creation-anding", sizeof(size_t), nbits);

	// Same, with 4 and operations
	s.clear();
	s2.clear();

	BENCH_START(fast_cr_anding4);
	for (size_t i = 0; i < nbits; i++) {
		const size_t r = random_bits[i];
		s2.set(r);
		s2.set(r+1);
		s2.set(r+2);
		s2.set(r+4);
		s &= s2;
		s &= s2;
		s &= s2;
		s &= s2;
		s2.clear();
	}
	BENCH_END_TRANSFORM(fast_cr_anding4, "fast-creation-anding-4", sizeof(size_t), nbits);

	return 0;
}
