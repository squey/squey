/**
 * \file open-save.h
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#ifndef _PICVIZ_OPEN_SAVE_H_
#define _PICVIZ_OPEN_SAVE_H_

#include <picviz/general.h>
#include <picviz/view.h>

#ifdef __cplusplus
 extern "C" {
#endif

LibPicvizDecl int picviz_save(picviz_view_t *view, char *filename);
LibPicvizDecl int picviz_open_inline(picviz_view_t *view, char *filename);
LibPicvizDecl picviz_view_t *picviz_open(char *filename);
LibPicvizDecl int picviz_open_is_picviz_type(char *filename);

#ifdef __cplusplus
 }
#endif

#endif /* _PICVIZ_OPEN_SAVE_H_ */
