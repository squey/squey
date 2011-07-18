#ifndef PVRUSH_PVTESTS_H
#define PVRUSH_PVTESTS_H

#include <pvcore/general.h>
#include <pvrush/PVExtractor.h>
#include <pvrush/PVSourceCreator.h>

#include <QString>

namespace PVRush
{

struct LibRushDecl PVTests
{
	/*! \brief Normalize a file with the given format
	 */
	static bool get_file_sc(PVFilter::PVArgument const& file, PVRush::PVFormat const& format, PVSourceCreator_p &sc);
};

}

#endif
