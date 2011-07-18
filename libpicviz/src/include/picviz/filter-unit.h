//! \file filter-unit.h
//! $Id: filter-unit.h 2490 2011-04-25 02:07:58Z psaade $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef _PICVIZ_FILTER_UNIT_H_
#define _PICVIZ_FILTER_UNIT_H_

#include <apr_general.h>

#include <picviz/general.h>

#define PICVIZ_FILTER_UNIT_MAXARGS 20

#ifdef __cplusplus
 extern "C" {
#endif



/******************************************************************************
 *
 * STRUCTURE
 *
 *****************************************************************************/

struct _picviz_filter_unit_argument_t {
	float a_float;
	char  a_char;
	char *a_char_p;
	int a_int;
	void *a_pointer;
	apr_uint64_t a_uint64;
};
typedef struct _picviz_filter_unit_argument_t picviz_filter_unit_argument_t;



struct _picviz_filter_unit_t {
	char *function_name;
	int function_index;

	int arguments_count;
	picviz_filter_unit_argument_t arguments[PICVIZ_FILTER_UNIT_MAXARGS];
};
typedef struct _picviz_filter_unit_t picviz_filter_unit_t;




/******************************************************************************
 ******************************************************************************
 *
 * CREATOR/DESTRUCTOR
 *
 ******************************************************************************
 *****************************************************************************/

LibPicvizDecl picviz_filter_unit_t *picviz_filter_unit_new(void);



/******************************************************************************
 ******************************************************************************
 *
 * ALL OTHER FUNCTIONS
 *
 ******************************************************************************
 *****************************************************************************/

LibPicvizDecl void picviz_filter_unit_argument_append(picviz_filter_unit_t *filter_unit, picviz_filter_unit_argument_t argument);
LibPicvizDecl void picviz_filter_unit_argument_append_char_p(picviz_filter_unit_t *filter_unit, char *arg);
LibPicvizDecl void picviz_filter_unit_argument_append_float(picviz_filter_unit_t *filter_unit, float arg);

LibPicvizDecl char picviz_filter_unit_argument_get_char(picviz_filter_unit_t *filter_unit, int position);
LibPicvizDecl char *picviz_filter_unit_argument_get_char_p(picviz_filter_unit_t *filter_unit, int position);
LibPicvizDecl float picviz_filter_unit_argument_get_float(picviz_filter_unit_t *filter_unit, int position);
LibPicvizDecl int picviz_filter_unit_argument_get_int(picviz_filter_unit_t *filter_unit, int position);
LibPicvizDecl void *picviz_filter_unit_argument_get_pointer(picviz_filter_unit_t *filter_unit, int position);
LibPicvizDecl apr_uint64_t picviz_filter_unit_argument_get_uint64(picviz_filter_unit_t *filter_unit, int position);



#ifdef __cplusplus
 }
#endif

#endif /* _PICVIZ_FILTER_UNIT_H_ */
