/**
 * @file
 *
 * @copyright (C) Picviz Labs 2013-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVPARALLELVIEW_PVZONERENDERINGBCI_TYPES_H
#define PVPARALLELVIEW_PVZONERENDERINGBCI_TYPES_H

#include <pvparallelview/PVZoneRendering_types.h>

#include <memory>

namespace PVParallelView
{

class PVZoneRenderingBCIBase;
typedef std::shared_ptr<PVZoneRenderingBCIBase> PVZoneRenderingBCIBase_p;

template <size_t Bbits>
class PVZoneRenderingBCI;

template <size_t Bbits>
using PVZoneRenderingBCI_p = std::shared_ptr<PVZoneRenderingBCI<Bbits>>;
} // namespace PVParallelView

#endif
