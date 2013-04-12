/**
 * \file stdint.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVCORE_STDINT_H
#define PVCORE_STDINT_H

// AG: commented as of boost 1.53.0, cstdint.hpp isn't compatible with NVCC
// Anyway, this was usefull w/ MSVC 2008, that didn't include stdint.h.
// I think this is ok now with MSVC 2010 !
#if 0
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

#ifdef __CUDACC__
#include <stdint.h>
#else
#include <cstdint>
#endif

#endif
