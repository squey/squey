/**
 * \file PVInput_types.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVKERNEL_RUSH_PVINPUT_TYPES_H
#define PVKERNEL_RUSH_PVINPUT_TYPES_H

#include <pvbase/export.h>

namespace PVRush {

// Forward declaration of PVInput
class LibKernelDecl PVInput;

// Pointer type to a PVInput
typedef boost::shared_ptr<PVInput> PVInput_p;

// Input offset
typedef uint64_t input_offset;

}

#endif
