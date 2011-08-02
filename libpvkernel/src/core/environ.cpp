/*
 * $Id: environ.cpp 3090 2011-06-09 04:59:46Z stricaud $
 * Copyright (C) Sebastien Tricaud 2010-2011
 * Copyright (C) Philippe Saade 2010-2011
 * 
 */

#include <QString>

#include <stdlib.h>

#include <pvcore/environ.h>

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


