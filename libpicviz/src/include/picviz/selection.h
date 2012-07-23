/**
 * \file selection.h
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#ifndef _PICVIZ_SELECTION_H_
#define _PICVIZ_SELECTION_H_

#include <picviz/general.h>

#include <stdint.h>

#ifdef __cplusplus
 extern "C" {
#endif


#define PICVIZ_SELECTION_CHUNK_SIZE 32
#define PICVIZ_SELECTION_NUMBER_OF_CHUNKS PICVIZ_LINES_MAX / PICVIZ_SELECTION_CHUNK_SIZE /* 262144*/
#define PICVIZ_SELECTION_NUMBER_OF_BYTES PICVIZ_LINES_MAX / 8

/* Selection for axes */
#define PICVIZ_SELECTION_AXES_NUMBER_OF_CHUNKS PICVIZ_AXES_MAX / PICVIZ_SELECTION_CHUNK_SIZE 
#define PICVIZ_SELECTION_AXES_NUMBER_OF_BYTES PICVIZ_AXES_MAX / 8

/* typedef void (*picviz_selection_function)(PVRow id, apr_array_header_t *, void *userdata); */


enum _picviz_selection_type_t {
	PICVIZ_SELECTION_TYPE_AXES,
	PICVIZ_SELECTION_TYPE_LINES,
};
typedef enum _picviz_selection_type_t picviz_selection_type_t;

/******************************************************************************
 *
 * STRUCTURE
 * the SELECTION structure
 *
 *****************************************************************************/
/*! Table Selection */
struct _picviz_selection_t {
	picviz_selection_type_t type;
	uint32_t *table;
	/* apr_uint32_t *table; /\*!< SelectionTable *\/ */
};
typedef struct _picviz_selection_t picviz_selection_t;

/******************************************************************************
 *
 * STRUCTURE
 * the TAG structure
 *
 *****************************************************************************/
/* Rajouter tableau de tags dans la scene */
struct _picviz_tag_t {
	char *name;
	picviz_selection_t *selection;
};


/******************************************************************************
 ******************************************************************************
 *
 * CREATOR/DESTRUCTOR
 *
 ******************************************************************************
 *****************************************************************************/

LibPicvizDecl picviz_selection_t *picviz_selection_new(void);
LibPicvizDecl void picviz_selection_destroy(picviz_selection_t *selection);




/******************************************************************************
 ******************************************************************************
 *
 * GET/SET functions
 *
 ******************************************************************************
 *****************************************************************************/

LibPicvizDecl int picviz_selection_get_line(picviz_selection_t *ts, pv_row lineid);
// LibPicvizDecl pv_row picviz_selection_get_line_index_of_nth_selected_line(picviz_selection_t *selection, pv_row n);
LibPicvizDecl int picviz_selection_get_number_of_selected_lines_in_range(picviz_selection_t *selection, pv_row a, pv_row b);

LibPicvizDecl void picviz_selection_set_line(picviz_selection_t *ts, pv_row lineid, int bool_value);




/******************************************************************************
 ******************************************************************************
 *
 * Generic functions (that do not act as Operators on selections).
 *
 *****************************************************************************
 *****************************************************************************/

// LibPicvizDecl pv_row picviz_selection_count(picviz_selection_t *selection, pv_row last_line_index);
/* LibPicvizDecl void picviz_selection_mapped_foreach(picviz_mapped_t *mapped, picviz_selection_t *selection, picviz_selection_function function, void *userdata); */
/* LibPicvizDecl void picviz_selection_nraw_foreach(picviz_source_t *source, picviz_selection_t *selection, picviz_selection_function function, void *userdata); */

/******************************************************************************
 ******************************************************************************
 *
 * the set of operators of type A2A : inplace on A
 *
 ******************************************************************************
 *****************************************************************************/

LibPicvizDecl void picviz_selection_A2A_inverse(picviz_selection_t *a);
LibPicvizDecl void picviz_selection_A2A_select_all(picviz_selection_t *a);
LibPicvizDecl void picviz_selection_A2A_select_even(picviz_selection_t *a);
LibPicvizDecl void picviz_selection_A2A_select_from_s_to_e(picviz_selection_t *a, int start, int end);
LibPicvizDecl void picviz_selection_A2A_select_none(picviz_selection_t *a);
LibPicvizDecl void picviz_selection_A2A_select_odd(picviz_selection_t *a);




/******************************************************************************
 ******************************************************************************
 *
 * the set of operators of type A2B : Operator(A, B) : A --> B
 *
 ******************************************************************************
 *****************************************************************************/

LibPicvizDecl void picviz_selection_A2B_copy(picviz_selection_t *a, picviz_selection_t *b);
LibPicvizDecl void picviz_selection_A2B_inverse(picviz_selection_t *a, picviz_selection_t *b);




/******************************************************************************
 ******************************************************************************
 *
 * the set of operators of type AB2A : (A operator B) --> A
 *
 ******************************************************************************
 *****************************************************************************/

LibPicvizDecl void picviz_selection_AB2A_and(picviz_selection_t *a, picviz_selection_t *b);
LibPicvizDecl void picviz_selection_AB2A_or(picviz_selection_t *a, picviz_selection_t *b);
LibPicvizDecl void picviz_selection_AB2A_substraction(picviz_selection_t *a, picviz_selection_t *b);
LibPicvizDecl void picviz_selection_AB2A_xor(picviz_selection_t *a, picviz_selection_t *b);




/******************************************************************************
 ******************************************************************************
 *
 * the set of operators of type AB2C : (A operator B) --> C
 *
 ******************************************************************************
 *****************************************************************************/

LibPicvizDecl void picviz_selection_AB2C_and(picviz_selection_t *a, picviz_selection_t *b, picviz_selection_t *c);
LibPicvizDecl void picviz_selection_AB2C_or(picviz_selection_t *a, picviz_selection_t *b, picviz_selection_t *c);
LibPicvizDecl void picviz_selection_AB2C_substraction(picviz_selection_t *a, picviz_selection_t *b, picviz_selection_t *c);
LibPicvizDecl void picviz_selection_AB2C_xor(picviz_selection_t *a, picviz_selection_t *b, picviz_selection_t *c);




#ifdef __cplusplus
 }
#endif

#endif /* _PICVIZ_SELECTION_H_ */
