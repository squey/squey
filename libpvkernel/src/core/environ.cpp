/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <QString>

#include <stdlib.h>

#include <pvkernel/core/environ.h>

int PVCore::mkenv(const char *name, const char *value)
{
  return setenv(name, value, 0);
}

int PVCore::rmenv(const char *name)
{
  return unsetenv(name);
}


