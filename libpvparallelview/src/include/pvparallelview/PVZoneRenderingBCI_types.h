#ifndef PVPARALLELVIEW_PVZONERENDERINGBCI_TYPES_H
#define PVPARALLELVIEW_PVZONERENDERINGBCI_TYPES_H

#include <pvparallelview/PVZoneRendering_types.h>

#include <memory>

namespace PVParallelView {

class PVZoneRenderingBCIBase;
typedef std::shared_ptr<PVZoneRenderingBCIBase> PVZoneRenderingBCIBase_p;

template <size_t Bbits>
class PVZoneRenderingBCI;

template <size_t Bbits>
using PVZoneRenderingBCI_p = std::shared_ptr<PVZoneRenderingBCI<Bbits>>;

}


#endif
