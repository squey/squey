/**
 * \file PVUtils.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVRUSH_PVUTILS_H
#define PVRUSH_PVUTILS_H

#include <QString>

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVAxesIndexType.h>
#include <pvkernel/rush/PVNraw.h>

namespace PVRush {
	namespace PVUtils {
		LibKernelDecl QString generate_key_from_axes_values(PVCore::PVAxesIndexType const& axes, PVRush::PVNraw::const_nraw_table_line const& values);
	}
}

#endif	/* PVRUSH_PVUTILS_H */
