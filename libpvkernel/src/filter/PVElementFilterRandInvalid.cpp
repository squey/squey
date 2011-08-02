#include <pvkernel/filter/PVElementFilterRandInvalid.h>
#include <cstdlib>
#include <stdio.h>

PVFilter::PVElementFilterRandInvalid::PVElementFilterRandInvalid() :
	PVElementFilter()
{
	INIT_FILTER_NOPARAM(PVElementFilterRandInvalid);
#ifdef WIN32
	std::srand(0); // Not a big deal, just for testing anyway
#else
	std::srand(time(NULL));
#endif
}

PVCore::PVElement& PVFilter::PVElementFilterRandInvalid::operator()(PVCore::PVElement &elt)
{
	bool invalidate = rand() & 1;
	if (invalidate)
		elt.set_invalid();
	return elt;
}

IMPL_FILTER_NOPARAM(PVFilter::PVElementFilterRandInvalid)
