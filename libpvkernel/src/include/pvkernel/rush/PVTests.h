/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVRUSH_PVTESTS_H
#define PVRUSH_PVTESTS_H

#include <pvkernel/rush/PVExtractor.h>
#include <pvkernel/rush/PVSourceCreator.h>

namespace PVRush
{

struct PVTests {
	/*! \brief Normalize a file with the given format
	 */
	static bool
	get_file_sc(PVInputDescription_p file, PVRush::PVFormat const& format, PVSourceCreator_p& sc);
};
}

#endif
