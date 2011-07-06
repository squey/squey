//! \file init.h
//! $Id: init.h 2489 2011-04-25 01:53:05Z psaade $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

/*
 * $Id: init.h 2489 2011-04-25 01:53:05Z psaade $
 * Copyright (C) Sebastien Tricaud 2010
 * Copyright (C) Philippe Saade 2010
 * 
 */

#ifndef _PICVIZ_INIT_H_
#define _PICVIZ_INIT_H_

#include <picviz/general.h>
/* #include <picviz/context.h> */

#ifdef __cplusplus
 extern "C" {
#endif

LibExport void picviz_init(int argc, char **argv);
LibExport void picviz_terminate(void);

#ifdef __cplusplus
 }
#endif

#endif /* _PICVIZ_INIT_H_ */
