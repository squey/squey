#ifndef PVPARALLELVIEW_PVZONERENDERINGBCI_TYPES_H
#define PVPARALLELVIEW_PVZONERENDERINGBCI_TYPES_H

#include <boost/shared_ptr.hpp>
#include <pvparallelview/PVZoneRendering_types.h>

namespace PVParallelView {

class PVZoneRenderingBCIBase;
typedef boost::shared_ptr<PVZoneRenderingBCIBase> PVZoneRenderingBCIBase_p;

template <size_t Bbits>
class PVZoneRenderingBCI;

template <size_t Bbits>
using PVZoneRenderingBCI_p = boost::shared_ptr<PVZoneRenderingBCI<Bbits>>;

}


#endif
