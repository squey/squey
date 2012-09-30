#ifndef PVCORE_STRINGTBB_H
#define PVCORE_STRINGTBB_H

#include <string>
#include <tbb/scalable_allocator.h>

namespace PVCore { namespace __impl {
size_t _Hash_bytes(const void* ptr, size_t len, size_t seed);
} }

// std::string with TBB's scalable allocator
namespace std {
typedef std::basic_string<char, std::char_traits<char>, tbb::scalable_allocator<char> > string_tbb;

template<>
class hash<string_tbb> {
public:
	inline size_t operator()(const string_tbb& s) const 
	{   
		return PVCore::__impl::_Hash_bytes(s.c_str(), s.size(), 0xc70f6907UL);
	}
};

}

#endif
