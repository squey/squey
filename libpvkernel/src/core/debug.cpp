/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
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

/** \file debug.c
 * \brief Helper functions to debug INENDI
 */

void debug_qstringlist(QStringList list)
{
	for (int i = 0; i < list.size(); ++i) {
		printf("%s", list.at(i).toLocal8Bit().constData());
		if (i + 1 < list.size())
			printf(",");
	}
	printf("\n");
}
