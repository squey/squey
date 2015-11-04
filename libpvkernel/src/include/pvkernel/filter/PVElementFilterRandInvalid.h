/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVFILTER_PVELEMENTFILTERRANDINVALID_H
#define PVFILTER_PVELEMENTFILTERRANDINVALID_H

#include <pvkernel/core/general.h>
#include <pvkernel/filter/PVElementFilter.h>

namespace PVFilter {

// This class will randomly invalidate elements
// This is used for the controller's test cases
class PVElementFilterRandInvalid : public PVElementFilter {
public:
	PVElementFilterRandInvalid();
public:
	PVCore::PVElement& operator()(PVCore::PVElement& elt);

	CLASS_FILTER_NONREG_NOPARAM(PVElementFilterRandInvalid)
};

}

#endif
