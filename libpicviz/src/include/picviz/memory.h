//! \file memory.h
//! $Id: memory.h 2490 2011-04-25 02:07:58Z psaade $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef _PICVIZ_MEMORY_H_
#define _PICVIZ_MEMORY_H_

#include <picviz/general.h>

#ifdef __cplusplus
 extern "C" {
#endif

LibExport void *picviz_malloc(size_t size);
LibExport void *picviz_realloc(void *ptr, size_t size);
LibExport void picviz_free(void *ptr);

#ifdef __cplusplus
 }
#endif

#endif	/* _PICVIZ_MEMORY_H_ */

