//! \file open-save.h
//! $Id: open-save.h 2490 2011-04-25 02:07:58Z psaade $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef _PICVIZ_OPEN_SAVE_H_
#define _PICVIZ_OPEN_SAVE_H_

#include <picviz/general.h>
#include <picviz/view.h>

#ifdef __cplusplus
 extern "C" {
#endif

LibExport int picviz_save(picviz_view_t *view, char *filename);
LibExport int picviz_open_inline(picviz_view_t *view, char *filename);
LibExport picviz_view_t *picviz_open(char *filename);
LibExport int picviz_open_is_picviz_type(char *filename);

#ifdef __cplusplus
 }
#endif

#endif /* _PICVIZ_OPEN_SAVE_H_ */
