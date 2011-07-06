//! \file square-area.h
//! $Id: square-area.h 2490 2011-04-25 02:07:58Z psaade $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef _PICVIZ_SQUARE_AREA_H_
#define _PICVIZ_SQUARE_AREA_H_

#include <picviz/general.h>

#ifdef __cplusplus
 extern "C" {
#endif


/******************************************************************************
 *
 * STRUCTURE
 *
 *****************************************************************************/

struct _picviz_square_area_t {
	float end_x;
	float end_y;
	float start_x;
	float start_y;
};
typedef struct _picviz_square_area_t picviz_square_area_t;



/******************************************************************************
 *
 * CREATOR/DESTRUCTOR
 *
 *****************************************************************************/

LibExport picviz_square_area_t *picviz_square_area_new(void);
LibExport void picviz_square_area_destroy(picviz_square_area_t *sa);



/******************************************************************************
 ******************************************************************************
 *
 * GET/SET functions
 *
 ******************************************************************************
 *****************************************************************************/

LibExport float picviz_square_area_get_end_x(picviz_square_area_t *sa);
LibExport float picviz_square_area_get_end_y(picviz_square_area_t *sa);
LibExport float picviz_square_area_get_start_x(picviz_square_area_t *sa);
LibExport float picviz_square_area_get_start_y(picviz_square_area_t *sa);


LibExport void picviz_square_area_set_end(picviz_square_area_t *sa, float ex, float ey);
LibExport void picviz_square_area_set_end_x(picviz_square_area_t *sa, float ex);
LibExport void picviz_square_area_set_end_y(picviz_square_area_t *sa, float ey);
LibExport void picviz_square_area_set_start(picviz_square_area_t *sa, float sx, float sy);
LibExport void picviz_square_area_set_start_x(picviz_square_area_t *sa, float sx);
LibExport void picviz_square_area_set_start_y(picviz_square_area_t *sa, float sy);






#ifdef __cplusplus
 }
#endif

#endif /* _PICVIZ_SQUARE_AREA_H_ */
