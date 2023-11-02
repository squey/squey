/* * MIT License
 *
 * © ESI Group, 2015
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
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
static const QColor frame_text_color(QRgb(0x838989));
static const QMargins frame_margins(14, 6, 14, 7);
static const QMargins frame_offsets(3, 2, 3, 2);

} // namespace PVParallelView

//#include <pvkernel/core/PVAllocators.h>

struct PVZoneID : std::pair<PVCol, PVCol> {
	using pair::pair;

	constexpr bool is_valid() const { return first != PVCol() and second != PVCol(); }
	constexpr bool is_invalid() const { return not is_valid(); }
};
static constexpr PVZoneID PVZONEID_INVALID = PVZoneID(PVCol(), PVCol());
static constexpr size_t PVZONEINDEX_INVALID = size_t(-1);

constexpr bool operator!=(PVZoneID const& lhs, PVZoneID const& rhs)
{
	return lhs.first != rhs.first || lhs.second != lhs.second;
}
constexpr bool operator==(PVZoneID const& lhs, PVZoneID const& rhs)
{
	return not(lhs != rhs);
}

inline std::ostream& operator<<(std::ostream& os, PVZoneID const& z)
{
	return os << z.first << ":" << z.second;
}

namespace std
{
template <>
struct hash<PVZoneID> {
	size_t operator()(PVZoneID const& c) const
	{
		return (int64_t(c.first) << 32) + int64_t(c.second);
	}
};
} // namespace std

Q_DECLARE_METATYPE(PVZoneID);
Q_DECLARE_METATYPE(size_t);

static bool _ __attribute((unused)) = []() {
	qRegisterMetaType<PVZoneID>("PVZoneID");
	qRegisterMetaType<size_t>("size_t");
	return true;
}();

static constexpr const int BCI_BUFFERS_COUNT = 10;

#endif
