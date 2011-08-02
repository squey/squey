#ifndef PVRUSH_PVTESTS_H
#define PVRUSH_PVTESTS_H

#include <pvkernel/core/general.h>
#include <pvkernel/rush/PVExtractor.h>
#include <pvkernel/rush/PVSourceCreator.h>

#include <QString>

namespace PVRush
{

struct LibKernelDecl PVTests
{
	/*! \brief Normalize a file with the given format
	 */
	static bool get_file_sc(PVCore::PVArgument const& file, PVRush::PVFormat const& format, PVSourceCreator_p &sc);
};

}

#endif
