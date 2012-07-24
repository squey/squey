/**
 * \file PVDirectory.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVCORE_PVDIRECTORY_H
#define PVCORE_PVDIRECTORY_H

#include <pvkernel/core/general.h>
#include <QString>

namespace PVCore {

class LibKernelDecl PVDirectory
{
public:
	static bool remove_rec(QString const& dirName);
	static QString temp_dir(QString const& pattern);
};

}

#endif
