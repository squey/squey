/**
 * \file nraw_sort_column.cpp
 *
 * Copyright (C) Picviz Labs 2012
 */

#include <pvkernel/rush/PVNrawDiskBackend.h>
#include <pvkernel/core/picviz_bench.h>
#include <pvkernel/core/PVUnicodeString.h>
#include <pvkernel/core/picviz_assert.h>

#include <tbb/enumerable_thread_specific.h>
#include <tbb/scalable_allocator.h>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>

#include <string>
#include <set>

#define MIN_SIZE 1
#define MAX_SIZE 256
#define N (10*(1<<20))

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

typedef std::pair<PVCore::PVUnicodeString, uint32_t> string_index_t;

struct MultimapCompare
{
	inline bool operator()(const PVCore::PVUnicodeString& s1, const PVCore::PVUnicodeString& s2) const
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


typedef std::multiset<string_index_t, MultisetCompare, tbb::scalable_allocator<string_index_t>> multiset_string_index_t;

class TBBMultisetReduce
{
public:
	TBBMultisetReduce(tbb::enumerable_thread_specific<multiset_string_index_t>& tls_multiset, const std::vector<multiset_string_index_t*>& multiset_pointers) : _tls_multiset(tls_multiset), _multiset_pointers(multiset_pointers) {}
	TBBMultisetReduce(TBBMultisetReduce o, tbb::split) : _tls_multiset(o._tls_multiset), _multiset_pointers(o._multiset_pointers) {}

public:
	void operator() (const tbb::blocked_range<uint32_t>& range) const
	{
		PV_ASSERT_VALID(range.size() == 1);
		printf("operator(): %p %d\n",  this, range.begin());
		_multiset = _multiset_pointers[range.begin()];
		_index = range.begin();
	}

	void join(TBBMultisetReduce& rhs)
	{
#if VERBOSE
		printf("merge between %d and %d\n", _index, rhs._index);
#endif
		for (auto pair : *rhs._multiset) {
			_multiset->insert(pair);
		}
	}

	multiset_string_index_t& get_reduced_multiset() { printf("get_reduced_multiset() for %d\n", _index); return *_multiset; }

private:
	tbb::enumerable_thread_specific<multiset_string_index_t>& _tls_multiset;
	const std::vector<multiset_string_index_t*>& _multiset_pointers;
	mutable multiset_string_index_t* _multiset = nullptr;
	mutable uint32_t _index = 0;
};

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
	// std::sort
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

#if INSERT_SORT
	std::multimap<PVCore::PVUnicodeString, uint32_t, MultimapCompare> multimap;

	// Multimap at() (fill cache)
	BENCH_START(multimap_first_at_run);
	for (size_t i = 0; i < N; i++) {
		size_t size;
		const char* field = backend.at(i, 0, size);
		PVCore::PVUnicodeString s(field, size);
		multimap.insert(std::pair<PVCore::PVUnicodeString, uint32_t>(s, i));
	}
	BENCH_END(multimap_first_at_run, "multimap_first_at_run", 1, 1, 1, total_column_size);
	multimap.clear();

	// Multimap visit_column2
	multimap.clear();
	BENCH_START(multimap_visit_column2);
		backend.visit_column2(0, [&](PVRow i, const char* buf, size_t size)
		{
			PVCore::PVUnicodeString s(buf, size);
			multimap.insert(std::pair<PVCore::PVUnicodeString, uint32_t>(s, i));
		});
	BENCH_END(multimap_visit_column2, "multimap_visit_column2", 1, 1, 1, total_column_size);
	#if VERBOSE
		for (auto val = multimap.begin(); val != multimap.end(); ++val) {
			printf("%.*s -> ", val->first.size(), val->first.buffer());
			printf("%d\n", val->second);
		}
	#endif

	// Multiset visit_column2
	std::multiset<std::pair<PVCore::PVUnicodeString, uint32_t>, MultisetCompare> multiset;
	std::multiset<std::pair<PVCore::PVUnicodeString, uint32_t>, MultisetCompare>::iterator it = multiset.begin();
	BENCH_START(multiset_visit_column2);
		backend.visit_column2(0, [&](PVRow i, const char* buf, size_t size)
		{
			PVCore::PVUnicodeString s(buf, size);
			/*it =*/ multiset.insert(/*it,*/ std::pair<PVCore::PVUnicodeString, uint32_t>(s, i));
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
	tbb::enumerable_thread_specific<multiset_string_index_t> tbb_multiset;
	BENCH_START(tbb_multiset_bench);
	backend.visit_column_tbb(0, [&tbb_multiset](size_t i, const char* buf, size_t size)
		{
			PVCore::PVUnicodeString s(buf, size);
			/*it =*/ tbb_multiset.local().insert(/*it,*/ std::pair<PVCore::PVUnicodeString, uint32_t>(s, i));
		});
	BENCH_END(tbb_multiset_bench, "tbb_multiset_bench", 1, 1, 1, total_column_size);

	// Serial multiset merge
	BENCH_START(serial_multiset_reduce);
	multiset_string_index_t merged_multiset;
	for (multiset_string_index_t& multiset : tbb_multiset) {
		for (string_index_t pair : multiset) {
			merged_multiset.insert(pair);
		}
	}
	BENCH_END(serial_multiset_reduce, "serial_multiset_reduce", 1, 1, 1, total_column_size);

	// Parallel multiset merge
	BENCH_START(tbb_multiset_reduce);
	std::vector<multiset_string_index_t*> multiset_pointers;
	multiset_pointers.reserve(tbb_multiset.size());
	int index = 0;
	for (multiset_string_index_t& multiset : tbb_multiset) {
		multiset_pointers[index++] = &multiset;
	}
	TBBMultisetReduce multiset_reduce(tbb_multiset, multiset_pointers);
	tbb::parallel_reduce(tbb::blocked_range<uint32_t>(0, tbb_multiset.size(), 1), multiset_reduce, tbb::simple_partitioner());
	BENCH_END(tbb_multiset_reduce, "tbb_multiset_reduce", 1, 1, 1, total_column_size);
	for (multiset_string_index_t& multiset : tbb_multiset) {
		printf("multiset.size()=%lu\n", multiset.size());
	}
#endif
}



