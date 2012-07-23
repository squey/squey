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
#ifdef WIN32
  QString str;
  str = QString(name) + QString("=") + QString(value);
  putenv(qPrintable(str));
#else
  return setenv(name, value, 0);
#endif

  return 0;
}

int PVCore::rmenv(const char *name)
{
#ifdef WIN32
  QString str;
  str = QString(name) + QString("=");
  putenv(qPrintable(str));
#else
  return unsetenv(name);
#endif

  return 0;
}


