/**
 * \file debug.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <iostream>
#include <fstream>
#include <string>

#include <QDateTime>
#include <QString>
#include <QStringList>

#include <stdarg.h>
#include <string.h>
#include <stdlib.h>

#include <pvkernel/core/debug.h>

/**
 * \ingroup PicvizMain
 * @{
 */

/** \file debug.c
 * \brief Helper functions to debug Picviz
 */

void debug_qstringlist(QStringList list)
{
	for (int i = 0; i < list.size(); ++i) {
		printf("%s", list.at(i).toLocal8Bit().constData());
		if (i+1 < list.size()) printf(",");
	}
	printf("\n");
}

/*@}*/

