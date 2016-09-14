/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVKERNEL_RUSH_PVINPUT_TYPES_H
#define PVKERNEL_RUSH_PVINPUT_TYPES_H

#include <memory>

namespace PVRush
{

// Forward declaration of PVInput
class PVInput;

// Pointer type to a PVInput
typedef std::shared_ptr<PVInput> PVInput_p;

// Input offset
typedef uint64_t input_offset;
} // namespace PVRush

#endif
