/*
 * $Id: environ.h 3090 2011-06-09 04:59:46Z stricaud $
 * Copyright (C) Sebastien Tricaud 2010-2011
 * Copyright (C) Philippe Saade 2010-2011
 * 
 */

#ifndef PVCORE_ENVIRON_H
#define PVCORE_ENVIRON_H

#include <pvkernel/core/general.h>

namespace PVCore {

	LibKernelDecl int mkenv(const char *name, const char *value);
	LibKernelDecl int rmenv(const char *name);

}

#endif	/* PVCORE_ENVIRON_H */
