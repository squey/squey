/**
 * \file environ.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
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


