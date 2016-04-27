/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVCORE_DEBUG_H
#define PVCORE_DEBUG_H

#include <QStringList>

#include <pvkernel/core/general.h>

/* #ifdef RELEASE */
/* #define PVCORE_DEBUG(f, ...) do { } while(0)  */
/* #else */
/* #define PVCORE_DEBUG(f, ...) do { PVCore::log(PVCore::debug, f, __VA_ARGS__) } while(0)  */
/* #endif */

void debug_qstringlist(QStringList list);

/* #define INENDI_ENTERING_DEBUG inendi_debug(INENDI_DEBUG_DEBUG, "==> Entering %s...\n",
 * __FUNCTION__); */
/* #define INENDI_LEAVING_DEBUG inendi_debug(INENDI_DEBUG_DEBUG, "<== Leaving %s...\n",
 * __FUNCTION__); */

#endif /* PVCORE_DEBUG_H */
