/**
 * \file nraw_sort_column.cpp
 *
 * Copyright (C) Picviz Labs 2012
 */

#include <pvkernel/rush/PVNrawDiskBackend.h>
#include <pvkernel/core/picviz_bench.h>
#include <pvkernel/core/PVUnicodeString.h>
#include <pvkernel/core/picviz_assert.h>
#include <pvkernel/core/PVHardwareConcurrency.h>

#include <tbb/enumerable_thread_specific.h>
#include <tbb/scalable_allocator.h>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>
#include <tbb/task_scheduler_init.h>

#include <string>
#include <set>
#include <algorithm>

#define MIN_SIZE 1
#define MAX_SIZE 256
#define N (1*(1<<20))

#define VERBOSE 0
#define STD_SORT 0
#define INSERT_SORT 1
#define PARALLEL_INSERT_SORT 1

PVRush::PVNrawDiskBackend backend;

bool sort_compare(uint32_t i, uint32_t j)
{
	size_t sizei;
	size_t sizej;
	const char* i1 = backend.at(i, 0, sizei);
	const char* j1 = backend.at(j, 0, sizej);
	PVCore::PVUnicodeString s1(i1, sizei);
	PVCore::PVUnicodeString s2(j1, sizej);

	bool res = s1.compare(s2) < 0;

	/*printf("%.*s ", sizei, i1);
	printf("%.*s ", sizej, j1);
	printf("%d\n", res);*/

	return res;
}

typedef std::pair<std::string_tbb, uint32_t> string_index_t;

struct MultimapCompare
{
	inline bool operator()(const std::string_tbb& s1, const std::string_tbb& s2) const
	{
		return s1.compare(s2) > 0;
	}
};

struct MultisetCompare
{
	inline bool operator()(const string_index_t& p1, const string_index_t& p2) const
	{
		int iret = p1.first.compare(p2.first);
		return iret > 0 || (iret == 0 && p1.second < p2.second);
	}
};

struct MultimapMultisetEqual
{
	inline bool operator()(const string_index_t& p1, const string_index_t& p2) const
	{
		return p1.first.compare(p2.first) == 0 && p1.second == p2.second;
	}
};

typedef std::multiset<string_index_t, MultisetCompare, tbb::scalable_allocator<string_index_t>> multiset_string_index_t;
typedef std::multimap<std::string_tbb, uint32_t, MultimapCompare, tbb::scalable_allocator<string_index_t>> multimap_string_index_t;
typedef tbb::enumerable_thread_specific<multiset_string_index_t> multiset_string_index_tls_t;

class TBBMultisetReduce
{
public:
	TBBMultisetReduce(const std::vector<multiset_string_index_t*>& multiset_pointers) : _multiset_pointers(multiset_pointers) {}
	TBBMultisetReduce(TBBMultisetReduce o, tbb::split) : _multiset_pointers(o._multiset_pointers) {}

public:
	void operator() (const tbb::blocked_range<uint32_t>& range)
	{
		PV_ASSERT_VALID(range.size() == 1);
#if VERBOSE
		//printf("operator(): %p %d\n",  this, range.begin());
#endif
		_index = range.begin();
		_multiset = _multiset_pointers[_index];
	}

	void join(TBBMultisetReduce& rhs)
	{
#if VERBOSE
		printf("merge between %d and %d\n", _index, rhs._index);
#endif
		for (const string_index_t& pair : *rhs._multiset) {
			_multiset->insert(std::move(pair));
		}
	}

	multiset_string_index_t& get_reduced_multiset() { printf("reduced multiset contains %lu values\n", _multiset->size()); return *_multiset; }

private:
	const std::vector<multiset_string_index_t*>& _multiset_pointers;
	multiset_string_index_t* _multiset = nullptr;
	uint32_t _index = 0;
};

void multiset_parallel_merge(multiset_string_index_tls_t& tbb_multiset, size_t total_column_size)
{
	// Parallel multiset merge
	BENCH_START(tbb_multiset_reduce);
	std::vector<multiset_string_index_t*> multiset_pointers;
	multiset_pointers.reserve(tbb_multiset.size());
	int index = 0;
	for (multiset_string_index_t& multiset : tbb_multiset) {
		multiset_pointers[index++] = &multiset;
	}
	TBBMultisetReduce multiset_reduce(multiset_pointers);
	tbb::parallel_deterministic_reduce(tbb::blocked_range<uint32_t>(0, tbb_multiset.size(), 1), multiset_reduce);
	BENCH_END(tbb_multiset_reduce, "tbb_multiset_reduce", 1, 1, 1, total_column_size);
#if VERBOSE
	for (multiset_string_index_t& multiset : tbb_multiset) {
		printf("multiset.size()=%lu\n", multiset.size());
	}
#endif
}

void multiset_serial_merge(multiset_string_index_tls_t& tbb_multiset, size_t total_column_size, multiset_string_index_t& merged_multiset)
{
	BENCH_START(serial_multiset_reduce);
	for (multiset_string_index_t& multiset : tbb_multiset) {
		for (const string_index_t& pair : multiset) {
			merged_multiset.insert(std::move(pair));
		}
	}
	BENCH_END(serial_multiset_reduce, "serial_multiset_reduce", 1, 1, 1, total_column_size);
}

bool compare_multimap_multiset()
{
	return true;
}

int main(int argc, char** argv)
{
	if (argc < 2) {
		std::cerr << "Usage: " << argv[0] << " path_nraw" << std::endl;
		return 1;
	}

	const char* nraw_path = argv[1];

	backend.init(nraw_path, 1);
	PVLOG_INFO("Writing NRAW...\n");
	char buf[MAX_SIZE];

	std::vector<uint32_t> random_vec;
	std::vector<uint32_t> vec;
	vec.reserve(N);
	random_vec.reserve(N);
	for (uint32_t i = 0; i < N; i++) {
		random_vec.push_back(65+(i%1000));
		vec.push_back(i);
	}
	std::random_shuffle(random_vec.begin(), random_vec.end());
#if VERBOSE
	printf("SHUFFLED_VEC:\n");
	for (size_t i = 0; i < N; i++) {
		printf("%c\n", random_vec[i]);
	}
#endif

	// NRAW insertions
	size_t total_column_size = 0;
	char big_string[200];
	const char A = 'A';
	memset((void*)big_string, A, sizeof(big_string));
	for (size_t i = 0; i < N; i++) {
		snprintf(buf, sizeof(buf), "%u" /*"%c"*/, random_vec[i]);
		strncat(buf, big_string, sizeof(big_string));
		size_t size = strlen(buf);
		backend.add(0, buf, size);
		total_column_size += size+1;
	}
	backend.flush();

	PVLOG_INFO("Serial sort... %d values\n", N);

#if STD_SORT
	// std::sort (random access to PVNrawDiskBackend)
	BENCH_START(std_sort);
	std::sort(vec.begin(), vec.end(), sort_compare);
	BENCH_END(std_sort, "std_sort", 1, 1, N, sizeof(uint32_t));
	#if VERBOSE
		printf("STD::SORT VEC:\n");
		for (size_t i = 0; i < N; i++) {
			printf("%d -> %d\n", i, vec[i]);
		}
	#endif
#endif


	/*const size_t nthreads = PVCore::PVHardwareConcurrency::get_physical_core_number();
	tbb::task_scheduler_init init(nthreads);*/

#if INSERT_SORT
	multimap_string_index_t multimap_ref;

	// Multimap at() (fill cache)
	BENCH_START(multimap_first_at_run);
	for (size_t i = 0; i < N; i++) {
		size_t size;
		const char* buf = backend.at(i, 0, size);
		std::string_tbb s(buf, size);
		multimap_ref.insert(std::move(string_index_t(s, i)));
	}
	BENCH_END(multimap_first_at_run, "multimap_first_at_run", 1, 1, 1, total_column_size);

	bool sorted = true;
	std::string_tbb previous = multimap_ref.begin()->first;
	for (const string_index_t& pair : multimap_ref) {
		previous = pair.first;
		sorted &= pair.first.compare(previous) <= 0;
	}
	std::cout << "sorted:" << std::boolalpha << sorted << std::endl;

	// Multimap visit_column2
	multimap_string_index_t multimap;
	BENCH_START(multimap_visit_column2);
		backend.visit_column2(0, [&](PVRow i, const char* buf, size_t size)
		{
			std::string_tbb s(buf, size);
			multimap.insert(std::move(string_index_t(s, i)));
		});
	BENCH_END(multimap_visit_column2, "multimap_visit_column2", 1, 1, 1, total_column_size);
	#if VERBOSE
		for (auto val = multimap.begin(); val != multimap.end(); ++val) {
			printf("%.*s -> ", val->first.size(), val->first.buffer());
			printf("%d\n", val->second);
		}
	#endif

	bool equal_to_ref =  (multimap == multimap_ref);
	std::cout << "equal to reference:" << std::boolalpha << equal_to_ref << std::endl;

	// Multiset visit_column2
	multiset_string_index_t multiset;
	multiset_string_index_t::iterator it = multiset.begin();
	BENCH_START(multiset_visit_column2);
		backend.visit_column2(0, [&](PVRow i, const char* buf, size_t size)
		{
			std::string_tbb s(buf, size);
			/*it =*/ multiset.insert(/*it,*/ std::move(string_index_t(s, i)));
		});
		/*// make a std::vector of indexes
		std::vector<uint32_t> index_vec;
		index_vec.reserve(N);
		int index = 0;
		for (auto val = multiset.rbegin(); val != multiset.rend(); ++val) {
			index_vec[val->second] = index;
			index++;
		}*/
	BENCH_END(multiset_visit_column2, "multiset_visit_column2", 1, 1, 1, total_column_size);
	#if VERBOSE
		for (auto val = multiset.begin(); val != multiset.end(); ++val) {
			printf("%.*s -> ", val->first.size(), val->first.buffer());
			printf("%d\n", val->second);
		}
	#endif
	equal_to_ref = (multimap_ref.size() == multiset.size()) && std::equal(multimap_ref.begin(), multimap_ref.end(), multiset.begin(), MultimapMultisetEqual());
	std::cout << "equal to reference:" << std::boolalpha << equal_to_ref << std::endl;

#endif

#if PARALLEL_INSERT_SORT
	uint32_t anti_optimization = 0;
	for (size_t i = 0; i < N; i++) {
		size_t size;
		uint32_t field = (uint64_t)backend.at(i, 0, size);
		anti_optimization &= field;
	}

	printf("%d\n", anti_optimization);

	// Parallel multiset sort
	multiset_string_index_tls_t tbb_multiset;
	BENCH_START(tbb_multiset_tls_bench);
	backend.visit_column_tbb(0, [&tbb_multiset](size_t i, const char* buf, size_t size)
		{
			std::string_tbb s(buf, size);
			/*it =*/ tbb_multiset.local().insert(/*it,*/ std::move(string_index_t(s, i)));
		});
	BENCH_END(tbb_multiset_tls_bench, "tbb_multiset_tls_bench", 1, 1, 1, total_column_size);

	// Serial multiset merge
	multiset_string_index_t serial_reduced_multiset;
	multiset_serial_merge(tbb_multiset, total_column_size, serial_reduced_multiset);
	//std::cout << "multimap_ref.size()=" << multimap_ref.size()<< " serial_reduced_multiset.size()" << serial_reduced_multiset.size() << std::endl;
	equal_to_ref = (multimap_ref.size() == serial_reduced_multiset.size()) && std::equal(multimap_ref.begin(), multimap_ref.end(), serial_reduced_multiset.begin(), MultimapMultisetEqual());
	std::cout << "equal to reference:" << std::boolalpha << equal_to_ref << std::endl;

	// Parallel multiset merge
	multiset_parallel_merge(tbb_multiset, total_column_size);
	const multiset_string_index_t& reduced_multiset = *tbb_multiset.begin();
	equal_to_ref = (multimap_ref.size() == reduced_multiset.size()) && std::equal(multimap_ref.begin(), multimap_ref.end(), reduced_multiset.begin(), MultimapMultisetEqual());
	std::cout << "equal to reference:" << std::boolalpha << equal_to_ref << std::endl;

#endif
}



