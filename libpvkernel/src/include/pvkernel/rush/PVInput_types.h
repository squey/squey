#ifndef PVKERNEL_RUSH_PVINPUT_TYPES_H
#define PVKERNEL_RUSH_PVINPUT_TYPES_H

#include <pvbase/export.h>

namespace PVRush {

// Forward declaration of PVInput
class LibKernelDecl PVInput;

// Pointer type to a PVInput
typedef boost::shared_ptr<PVInput> PVInput_p;

}

#endif
