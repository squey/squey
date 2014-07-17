/**
 * \file PVUtils.h
 *
 * Copyright (C) Picviz Labs 2010-2014
 */

#ifndef PVRUSH_PVUTILS_H
#define PVRUSH_PVUTILS_H

class QString;
#include <QByteArray>

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVAxesIndexType.h>
#include <pvkernel/rush/PVNraw.h>

#include <string.h>

namespace PVRush {
namespace PVUtils {
	//LibKernelDecl QString generate_key_from_axes_values(PVCore::PVAxesIndexType const& axes, PVRush::PVNraw::const_nraw_table_line const& values);

	const QByteArray get_file_checksum(const QString& path);
	bool files_have_same_content(const QString& path1, const QString& path2);

}
}

#endif	/* PVRUSH_PVUTILS_H */
