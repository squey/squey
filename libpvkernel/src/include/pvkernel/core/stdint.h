#ifndef PVCORE_STDINT_H
#define PVCORE_STDINT_H

#include <boost/cstdint.hpp>

using boost::int8_t;
using boost::int16_t;
using boost::int32_t;
using boost::int64_t;

using boost::uint8_t;
using boost::uint16_t;
using boost::uint32_t;
using boost::uint64_t;

#ifdef _MSC_VER
typedef unsigned __int64 size_t;
typedef __int64 ssize_t;
#endif

#endif
