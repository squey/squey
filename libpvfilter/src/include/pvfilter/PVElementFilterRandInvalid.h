#ifndef PVFILTER_PVELEMENTFILTERRANDINVALID_H
#define PVFILTER_PVELEMENTFILTERRANDINVALID_H

#include <pvcore/general.h>
#include <pvfilter/PVElementFilter.h>

namespace PVFilter {

// This class will randomly invalidate elements
// This is used for the controller's test cases
class LibExport PVElementFilterRandInvalid : public PVElementFilter {
public:
	PVElementFilterRandInvalid();
public:
	virtual PVCore::PVElement& operator()(PVCore::PVElement& elt);

	CLASS_FILTER(PVFilter::PVElementFilterRandInvalid)
};

}

#endif
