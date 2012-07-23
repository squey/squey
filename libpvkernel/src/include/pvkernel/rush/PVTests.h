/**
 * \file PVTests.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVRUSH_PVTESTS_H
#define PVRUSH_PVTESTS_H

#include <pvkernel/core/general.h>
#include <pvkernel/rush/PVExtractor.h>
#include <pvkernel/rush/PVSourceCreator.h>

namespace PVRush
{

struct LibKernelDecl PVTests
{
	/*! \brief Normalize a file with the given format
	 */
	static bool get_file_sc(PVInputDescription_p file, PVRush::PVFormat const& format, PVSourceCreator_p &sc);
};

}

#endif
