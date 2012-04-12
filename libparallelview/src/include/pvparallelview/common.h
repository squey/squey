#ifndef PVPARALLELVIEW_CONFIG_H
#define PVPARALLELVIEW_CONFIG_H

#include <vector>
#include <stdint.h>

#define NBITS_INDEX 10
#define NBUCKETS ((1UL<<(2*NBITS_INDEX)))
#define MASK_INT_YCOORD (((1UL)<<(NBITS_INDEX))-1)

#define IMAGE_HEIGHT (1024)
#define MASK_INT_PLOTTED (~(1UL<<(32-NBITS_INDEX))-1)

#include <pvkernel/core/PVAllocators.h>

typedef std::vector<uint32_t, PVCore::PVAlignedAllocator<uint32_t, 16> > plotted_int_t;

#endif
