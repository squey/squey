/**
 * \file PVBCIBackendImage_types.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVPARALLELVIEW_PVBCIBACKENDIMAGETYPES_H
#define PVPARALLELVIEW_PVBCIBACKENDIMAGETYPES_H

#include <boost/shared_ptr.hpp>

namespace PVParallelView {

template <size_t Bbits>
class PVBCIBackendImage;

template <size_t Bbits>
using PVBCIBackendImage_p = boost::shared_ptr<PVBCIBackendImage<Bbits> >;

}

#endif
