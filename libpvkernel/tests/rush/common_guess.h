#ifndef LIBPVKERNEL_RUSH_TESTS_COMMON_GUESS_H
#define LIBPVKERNEL_RUSH_TESTS_COMMON_GUESS_H

#include <pvkernel/filter/PVFieldsFilter.h>

namespace pvtest
{

PVFilter::PVFieldsSplitter_p guess_filter(const char* filename, PVCol& axes_count);
}

#endif // LIBPVKERNEL_RUSH_TESTS_COMMON_GUESS_H
