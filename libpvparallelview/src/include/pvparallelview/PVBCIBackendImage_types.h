/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVPARALLELVIEW_PVBCIBACKENDIMAGETYPES_H
#define PVPARALLELVIEW_PVBCIBACKENDIMAGETYPES_H

#include <memory>

namespace PVParallelView {

class PVBCIBackendImage;
typedef std::shared_ptr<PVBCIBackendImage> PVBCIBackendImage_p;

//template <size_t Bbits>
//using PVBCIBackendImage_p = std::shared_ptr<PVBCIBackendImage<Bbits> >;

}

#endif
