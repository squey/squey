#ifndef PVFILTER_PVELEMENTFILTERRANDINVALID_H
#define PVFILTER_PVELEMENTFILTERRANDINVALID_H

#include <pvkernel/core/general.h>
#include <pvkernel/filter/PVElementFilter.h>

namespace PVFilter {

// This class will randomly invalidate elements
// This is used for the controller's test cases
class LibKernelDecl PVElementFilterRandInvalid : public PVElementFilter {
public:
	PVElementFilterRandInvalid();
public:
	virtual PVCore::PVElement& operator()(PVCore::PVElement& elt);

	CLASS_FILTER(PVFilter::PVElementFilterRandInvalid)
};

}

#endif
