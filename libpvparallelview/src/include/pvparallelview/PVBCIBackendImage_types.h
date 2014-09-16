/**
 * \file PVBCIBackendImage_types.h
 *
 * Copyright (C) Picviz Labs 2010-2012
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
