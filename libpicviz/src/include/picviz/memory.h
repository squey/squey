/**
 * \file memory.h
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#ifndef _PICVIZ_MEMORY_H_
#define _PICVIZ_MEMORY_H_

#include <picviz/general.h>

#ifdef __cplusplus
 extern "C" {
#endif

LibPicvizDecl void *picviz_malloc(size_t size);
LibPicvizDecl void *picviz_realloc(void *ptr, size_t size);
LibPicvizDecl void picviz_free(void *ptr);

#ifdef __cplusplus
 }
#endif

#endif	/* _PICVIZ_MEMORY_H_ */

