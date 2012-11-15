/**
 * \file common.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVPARALLELVIEW_CONFIG_H
#define PVPARALLELVIEW_CONFIG_H

#include <vector>
#include <stdint.h>
#include <QMetaType>

#include <boost/integer/static_log2.hpp>

#define NBITS_INDEX 10
#define NBUCKETS ((1UL<<(2*NBITS_INDEX)))

// the cardinal of buckets ranges
#define BUCKET_ELT_COUNT (1 << (32 - NBITS_INDEX))

#define PARALLELVIEW_ZT_BBITS  10
#define PARALLELVIEW_ZZT_BBITS 11

#define PARALLELVIEW_ZOOM_WIDTH 512

#define PARALLELVIEW_MAX_BCI_CODES ((1UL<<(2*PARALLELVIEW_ZZT_BBITS))+2*(1UL<<(PARALLELVIEW_ZZT_BBITS))*(PARALLELVIEW_ZOOM_WIDTH))

#if (NBUCKETS % 2 != 0)
#error NBUCKETS must be a multiple of 2
#endif

#define MASK_INT_YCOORD (((1UL)<<NBITS_INDEX)-1)

#define IMAGE_HEIGHT (1024)
#define IMAGE_WIDTH (2048)
#define PARALLELVIEW_IMAGE_HEIGHT IMAGE_HEIGHT
#define PARALLELVIEW_IMAGE_WIDTH IMAGE_WIDTH

#define PARALLELVIEW_AXIS_WIDTH 3

#if (PARALLELVIEW_AXIS_WIDTH == 1)
#error PARALLELVIEW_AXIS_WIDTH must be strictly greater than 1
#endif

#if (PARALLELVIEW_AXIS_WIDTH % 2 != 1)
#error PARALLELVIEW_AXIS_WIDTH must be odd
#endif

// psaade : next value should be 128 according to aguinet, for the moment
#define PARALLELVIEW_ZONE_MIN_WIDTH 16
#define PARALLELVIEW_BASE_ZONE_WIDTH 64
#define PARALLELVIEW_ZONE_DEFAULT_WIDTH 256
#define PARALLELVIEW_ZONE_MAX_WIDTH 1024
#define PARALLELVIEW_MAX_DRAWN_ZONES 30

#ifndef __CUDACC__
static_assert((1<<(boost::static_log2<PARALLELVIEW_ZONE_MIN_WIDTH>::value) == PARALLELVIEW_ZONE_MIN_WIDTH), "Must be a power of two");
static_assert((1<<(boost::static_log2<PARALLELVIEW_BASE_ZONE_WIDTH>::value) == PARALLELVIEW_BASE_ZONE_WIDTH), "Must be a power of two");
static_assert((1<<(boost::static_log2<PARALLELVIEW_ZONE_MAX_WIDTH>::value) == PARALLELVIEW_ZONE_MAX_WIDTH), "Must be a power of two");
#endif

#define MASK_INT_PLOTTED (~(1UL<<(32-NBITS_INDEX))-1)

namespace PVParallelView {

enum {
	AxisWidth = PARALLELVIEW_AXIS_WIDTH,
	ImageHeight = PARALLELVIEW_IMAGE_HEIGHT,
	ImageWidth = PARALLELVIEW_IMAGE_WIDTH,
	ZoneMinWidth = PARALLELVIEW_ZONE_MIN_WIDTH,
	ZoneMaxWidth = PARALLELVIEW_ZONE_MAX_WIDTH,
	ZoneDefaultWidth = PARALLELVIEW_ZONE_DEFAULT_WIDTH,
	MaxDrawnZones = PARALLELVIEW_MAX_DRAWN_ZONES,
	MaxBciCodes = PARALLELVIEW_MAX_BCI_CODES
};


#ifdef __CUDACC__
// nvcc does not support C++0x !
#define CUDA_CONSTEXPR const
#else
#define CUDA_CONSTEXPR constexpr
#endif

template <size_t Bbits>
struct constants
{
	CUDA_CONSTEXPR static uint32_t image_height = ((uint32_t)1)<<Bbits;
	CUDA_CONSTEXPR static uint32_t mask_int_ycoord = (((uint32_t)1)<<Bbits)-1;
};

}

#include <pvkernel/core/PVAllocators.h>

typedef PVCol PVZoneID;
#define PVZONEID_INVALID (-1)

Q_DECLARE_METATYPE(PVZoneID);

#define BCI_BUFFERS_COUNT 10

#endif
