/**
 * @file
 *
 * @copyright (C) Picviz Labs 2013-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVPARALLELVIEW_PVHITGRAPHDATA_H
#define PVPARALLELVIEW_PVHITGRAPHDATA_H

#include <pvparallelview/PVHitGraphDataOMP.h>

namespace PVParallelView
{

// Choose which implementation must be used.
typedef PVHitGraphDataOMP PVHitGraphData;
} // namespace PVParallelView

#endif
