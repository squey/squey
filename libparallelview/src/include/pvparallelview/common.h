#ifndef PVPARALLELVIEW_CONFIG_H
#define PVPARALLELVIEW_CONFIG_H

#include <vector>
#include <stdint.h>

#define NBITS_INDEX 10
#define NBUCKETS ((1UL<<(2*NBITS_INDEX)))

#if (NBUCKETS % 2 != 0)
#error NBUCKETS must be a multiple of 2
#endif

#define MASK_INT_YCOORD (((1UL)<<(NBITS_INDEX))-1)

#define IMAGE_HEIGHT (1024)
#define PARALLELVIEW_IMAGE_HEIGHT IMAGE_HEIGHT

#define PARALLELVIEW_AXIS_WIDTH 3

#define MASK_INT_PLOTTED (~(1UL<<(32-NBITS_INDEX))-1)

namespace PVParallelView {

enum {
	AxisWidth = PARALLELVIEW_AXIS_WIDTH,
	ImageHeight = PARALLELVIEW_IMAGE_HEIGHT
};

}

#include <pvkernel/core/PVAllocators.h>

typedef PVCol PVZoneID;

#endif
