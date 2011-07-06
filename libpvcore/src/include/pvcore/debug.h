
/*
 * $Id: debug.h 3090 2011-06-09 04:59:46Z stricaud $
 * Copyright (C) Sebastien Tricaud 2010-2011
 * Copyright (C) Philippe Saade 2010-2011
 * 
 */

#ifndef PVCORE_DEBUG_H
#define PVCORE_DEBUG_H

#include <QStringList>

#include <pvcore/general.h>

/* #ifdef RELEASE */
/* #define PVCORE_DEBUG(f, ...) do { } while(0)  */
/* #else */
/* #define PVCORE_DEBUG(f, ...) do { PVCore::log(PVCore::debug, f, __VA_ARGS__) } while(0)  */
/* #endif */

LibExport void debug_qstringlist(QStringList list);

/* #define PICVIZ_ENTERING_DEBUG picviz_debug(PICVIZ_DEBUG_DEBUG, "==> Entering %s...\n", __FUNCTION__); */
/* #define PICVIZ_LEAVING_DEBUG picviz_debug(PICVIZ_DEBUG_DEBUG, "<== Leaving %s...\n", __FUNCTION__); */

#endif	/* PVCORE_DEBUG_H */
