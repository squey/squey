/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVPARALLELVIEW_CONFIG_H
#define PVPARALLELVIEW_CONFIG_H

#include <vector>
#include <pvbase/types.h>

#include <QMetaType>
#include <QColor>
#include <QMargins>

#include <boost/integer/static_log2.hpp>

static constexpr const int NBITS_INDEX = 10;
static constexpr const int NBUCKETS = ((1UL << (2 * NBITS_INDEX)));

// the cardinal of buckets ranges
static constexpr const int BUCKET_ELT_COUNT = (1 << (32 - NBITS_INDEX));

static constexpr const int PARALLELVIEW_ZT_BBITS = 10;
static constexpr const int PARALLELVIEW_ZZT_BBITS = 11;

static constexpr const int PARALLELVIEW_ZOOM_WIDTH = 512;

static constexpr const int PARALLELVIEW_MAX_BCI_CODES =
    ((1UL << (2 * PARALLELVIEW_ZZT_BBITS)) +
     2 * (1UL << (PARALLELVIEW_ZZT_BBITS)) * (PARALLELVIEW_ZOOM_WIDTH));

static_assert(NBUCKETS % 2 == 0, "NBUCKETS must be a multiple of 2");

static constexpr const int MASK_INT_YCOORD = (((1UL) << NBITS_INDEX) - 1);

static constexpr const int IMAGE_HEIGHT = 1024;
static constexpr const int IMAGE_WIDTH = 2048;
static constexpr const int PARALLELVIEW_IMAGE_HEIGHT = IMAGE_HEIGHT;
static constexpr const int PARALLELVIEW_IMAGE_WIDTH = IMAGE_WIDTH;

static constexpr const int PARALLELVIEW_AXIS_WIDTH = 3;

static_assert(PARALLELVIEW_AXIS_WIDTH != 1,
              "PARALLELVIEW_AXIS_WIDTH must be strictly greater than 1");

static_assert(PARALLELVIEW_AXIS_WIDTH % 2 == 1, "PARALLELVIEW_AXIS_WIDTH must be odd");

// psaade : next value should be 128 according to aguinet, for the moment
static constexpr const int PARALLELVIEW_ZONE_MIN_WIDTH = 16;
static constexpr const int PARALLELVIEW_ZONE_BASE_WIDTH = 64;
static constexpr const int PARALLELVIEW_ZONE_DEFAULT_WIDTH = 256;
static constexpr const int PARALLELVIEW_ZONE_MAX_WIDTH = 1024;
static constexpr const int PARALLELVIEW_MAX_DRAWN_ZONES = 30;

static_assert((1 << (boost::static_log2<PARALLELVIEW_ZONE_MIN_WIDTH>::value) ==
               PARALLELVIEW_ZONE_MIN_WIDTH),
              "Must be a power of two");
static_assert((1 << (boost::static_log2<PARALLELVIEW_ZONE_BASE_WIDTH>::value) ==
               PARALLELVIEW_ZONE_BASE_WIDTH),
              "Must be a power of two");
static_assert((1 << (boost::static_log2<PARALLELVIEW_ZONE_MAX_WIDTH>::value) ==
               PARALLELVIEW_ZONE_MAX_WIDTH),
              "Must be a power of two");

namespace PVParallelView
{

enum {
	AxisWidth = PARALLELVIEW_AXIS_WIDTH,
	ImageHeight = PARALLELVIEW_IMAGE_HEIGHT,
	ImageWidth = PARALLELVIEW_IMAGE_WIDTH,
	ZoneMinWidth = PARALLELVIEW_ZONE_MIN_WIDTH,
	ZoneMaxWidth = PARALLELVIEW_ZONE_MAX_WIDTH,
	ZoneDefaultWidth = PARALLELVIEW_ZONE_DEFAULT_WIDTH,
	ZoneBaseWidth = PARALLELVIEW_ZONE_BASE_WIDTH,
	MaxDrawnZones = PARALLELVIEW_MAX_DRAWN_ZONES,
	MaxBciCodes = PARALLELVIEW_MAX_BCI_CODES
};

template <size_t Bbits>
struct constants {
	static const constexpr size_t image_height = ((uint32_t)1) << Bbits;
	static const constexpr size_t mask_int_ycoord = (((uint32_t)1) << Bbits) - 1;
};

/* common views information/stats constants
 */
static const QColor frame_bg_color(0xff, 0xfe, 0xee, 0xdc);
static const QString frame_qss_bg_color("background-color: " +
                                        frame_bg_color.name(QColor::HexArgb) + ";");
static const QColor frame_text_color(0x40, 0x40, 0x40);
static const QMargins frame_margins(14, 6, 14, 7);
static const QMargins frame_offsets(3, 2, 3, 2);

} // namespace PVParallelView

//#include <pvkernel/core/PVAllocators.h>

using PVZoneID = PVCol::value_type;
static const PVZoneID PVZONEID_INVALID = PVCol::INVALID_VALUE;

Q_DECLARE_METATYPE(PVZoneID);

DEFINE_STRONG_TYPEDEF(PVZoneIDOffset, unsigned int)

static constexpr const int BCI_BUFFERS_COUNT = 10;

#endif
