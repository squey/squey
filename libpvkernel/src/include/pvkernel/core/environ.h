/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVCORE_ENVIRON_H
#define PVCORE_ENVIRON_H

#include <pvkernel/core/general.h>

namespace PVCore
{

int mkenv(const char* name, const char* value);
int rmenv(const char* name);
}

#endif /* PVCORE_ENVIRON_H */
