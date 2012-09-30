#include <string>
#include <functional>

#include <tbb/enumerable_thread_specific.h>
#include <tbb/scalable_allocator.h>

/*unsigned int qHash(const std::string& s)
{
	//return (unsigned int)std::hash<std::string>()(s);
	switch (s.size()) {
		case 0:
			return 0;
		case 1:
			return s[0];
		case 2:
			return *((const uint16_t*)s.c_str());
		case 3:
			return (uint32_t) s[0] | ((uint32_t)s[1]<<8) | ((uint32_t)s[2]<<16);
		default:
			return *((const uint32_t*)s.c_str());
	}
	return 0;
}*/

typedef std::basic_string<char, std::char_traits<char>, tbb::scalable_allocator<char> > my_str;

inline std::size_t
unaligned_load(const char* p)
{
	std::size_t result;
	__builtin_memcpy(&result, p, sizeof(result));
	return result;
}

// Loads n bytes, where 1 <= n < 8.
inline std::size_t
load_bytes(const char* p, int n)
{
	std::size_t result = 0;
	--n;
	do  
		result = (result << 8) + static_cast<unsigned char>(p[n]);
	while (--n >= 0); 
	return result;
}

inline std::size_t
shift_mix(std::size_t v)
{ return v ^ (v >> 47);}

// Implementation of Murmur hash for 64-bit size_t.
size_t
_Hash_bytes(const void* ptr, size_t len, size_t seed)
{
	static const size_t mul = (0xc6a4a793UL << 32UL) + 0x5bd1e995UL;
	const char* const buf = static_cast<const char*>(ptr);

	// Remove the bytes not divisible by the sizeof(size_t).  This
	// allows the main loop to process the data as 64-bit integers.
	const int len_aligned = len & ~0x7;
	const char* const end = buf + len_aligned;
	size_t hash = seed ^ (len * mul);
	for (const char* p = buf; p != end; p += 8)
	{
		const size_t data = shift_mix(unaligned_load(p) * mul) * mul;
		hash ^= data;
		hash *= mul;
	}
	if ((len & 0x7) != 0)
	{
		const size_t data = load_bytes(end, len & 0x7);
		hash ^= data;
		hash *= mul;
	}
	hash = shift_mix(hash) * mul;
	hash = shift_mix(hash);
	return hash;
}

namespace std {
	template<>
		class hash<my_str> {
			public:
				inline size_t operator()(const my_str &s) const 
				{
					return _Hash_bytes(s.c_str(), s.size(), 0xc70f6907UL);
				}
		};
}

inline unsigned int qHash(const std::string& s)
{
	return (unsigned int)std::hash<std::string>()(s);
}

inline unsigned int qHash(const my_str& s)
{
	return (unsigned int)std::hash<my_str>()(s);
}


#include <pvkernel/core/picviz_assert.h>
#include <pvkernel/rush/PVNrawDiskBackend.h>
#include <pvkernel/core/picviz_bench.h>
#include <pvkernel/core/PVUnicodeString.h>

#include <iostream>
#include <sstream>

#include <tbb/tick_count.h>

#include <QSet>
#include <unordered_set>

#define MIN_SIZE 1
#define MAX_SIZE 256
#define N (100*(1<<20))
#define LATENCY_N 100000

#include <set>

#define NBYTES_IDX
size_t get_buf_size(size_t i)
{
	return (i%(MAX_SIZE-MIN_SIZE+1))+MIN_SIZE;
	//return 10;
}

uint16_t compute_uni_hash(PVCore::PVUnicodeString const& s)
{
	switch (s.size()) {
		case 0:
			return 0;
		case 1:
			return s.buffer()[0];
		default:
			const uint8_t* buf = s.buffer();
			return *((uint16_t*)buf);
	}
	return 0;
}

int main(int argc, char** argv)
{
	if (argc < 2) {
		std::cerr << "Usage: " << argv[0] << " path_nraw" << std::endl;
		return 1;
	}

	const char* nraw_path = argv[1];
	PVRush::PVNrawDiskBackend backend;
	backend.init(nraw_path, 2);

	PVLOG_INFO("Writing NRAW...\n");
	char buf[MAX_SIZE];
	size_t stotal = 0;
	for (size_t i = 0; i < N; i++) {
		const size_t sbuf = (rand()%(MAX_SIZE-MIN_SIZE+1))+MIN_SIZE;
		stotal += sbuf;
		memset(buf, 'a' + i%26, sbuf);
		backend.add(0, buf, sbuf);
	}
	backend.flush();

	PVLOG_INFO("Serial sort|uniq...\n");

#if 0
	{
		std::set<std::string> str_set;
		BENCH_START(set);
		backend.visit_column2(0, [&str_set](size_t, const char* buf, size_t n)
				{
					std::string new_s(buf, n);
					str_set.insert(new_s);
				});
		BENCH_END(set, "set", sizeof(char), stotal, 1, 1);
	}

	{
		std::set<std::string> str_set;
		BENCH_START(set);
		backend.visit_column2(0, [&str_set](size_t, const char* buf, size_t n)
				{
					std::string new_s(buf, n);
					str_set.insert(new_s);
				});
		BENCH_END(set, "set", sizeof(char), stotal, 1, 1);
	}

	std::unordered_set<std::string> str_uset;
	BENCH_START(uset);
	backend.visit_column2(0, [&str_uset](size_t, const char* buf, size_t n)
			{
				std::string new_s(buf, n);
				str_uset.insert(new_s);
			});
	BENCH_END(uset, "uset", sizeof(char), stotal, 1, 1);

	QSet<std::string> str_qset;
	str_qset.reserve(N);
	BENCH_START(qset);
	backend.visit_column2(0, [&str_qset](size_t, const char* buf, size_t n)
			{
				std::string new_s(buf, n);
				str_qset.insert(new_s);
			});
	BENCH_END(qset, "qset", sizeof(char), stotal, 1, 1);
#endif

	tbb::enumerable_thread_specific<QSet<my_str> > tbb_qset([=]{ QSet<my_str> ret; ret.reserve(N/12); return ret; });
	BENCH_START(qset_tbb);
	backend.visit_column_tbb(0, [&tbb_qset](size_t, const char* buf, size_t n)
			{
				my_str new_s(buf, n);
				tbb_qset.local().insert(new_s);
			});
	BENCH_END(qset_tbb, "qset-tbb", sizeof(char), stotal, 1, 1);
	BENCH_START(qset_tbb_red);
	typename decltype(tbb_qset)::iterator it_tls = tbb_qset.begin();
	QSet<my_str>& final = *it_tls;
	it_tls++;
	for (; it_tls != tbb_qset.end(); it_tls++) {
		final.unite(*it_tls);
	}
	BENCH_END(qset_tbb_red, "qset-tbb-red", sizeof(char), stotal, 1, 1);

#if 0
	std::vector<PVCore::PVUnicodeString> vec_uni;
	vec_uni.resize(65536, PVCore::PVUnicodeString((const uint8_t*) nullptr, 0));

	BENCH_START(uni_key);
	backend.visit_column2(0, [&vec_uni](size_t, const char* buf, size_t n)
			{
				PVCore::PVUnicodeString new_uni(buf, n);
				const uint16_t key = compute_uni_hash(new_uni);
				PVCore::PVUnicodeString& uni_str = vec_uni[key];
				if (uni_str.buffer() == 0) {
					char* new_buf = (char*) malloc(n);
					memcpy(new_buf, buf, n);
					uni_str = PVCore::PVUnicodeString(new_buf, n);
				}
				else {
					if (new_uni.compare(uni_str) < 0) {
						if (uni_str.size() >= new_uni.size()) {
							memcpy((void*) uni_str.buffer(), buf, n);
							uni_str = PVCore::PVUnicodeString(uni_str.buffer(), n);
						}
						else {
							free((void*) uni_str.buffer());
							char* new_buf = (char*) malloc(n);
							memcpy(new_buf, buf, n);
							uni_str = PVCore::PVUnicodeString(new_buf, n);
						}
					}
				}
			});
	BENCH_END(uni_key, "uni-key", sizeof(char), stotal, 1, 1);

	for (PVCore::PVUnicodeString const& s: vec_uni) {
		if (s.buffer() != nullptr) {
			fwrite(s.buffer(), 1, s.size(), stdout);
			printf("\n");
		}
	}
#endif

	return 0;
}
