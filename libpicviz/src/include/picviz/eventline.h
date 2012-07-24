/**
 * \file eventline.h
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#ifndef _PICVIZ_EVENTLINE_H_
#define _PICVIZ_EVENTLINE_H_


#include <picviz/selection.h>

#ifdef __cplusplus
 extern "C" {
#endif



/******************************************************************************
 *
 * STRUCTURE
 *
 *****************************************************************************/

struct _picviz_eventline_t {
	void *dtri;
	void *parent;

	int row_count;
	
	int first_index;
	int current_index;
	int last_index;
	
};
typedef struct _picviz_eventline_t picviz_eventline_t;


/******************************************************************************
 ******************************************************************************
 *
 * CREATOR/DESTRUCTOR
 *
 ******************************************************************************
 *****************************************************************************/

/**
 * Create a new eventline object.
 * A View possesses a single EventLine. That EventLine holds the three following information :
 *  - the first element that can be plotted : EventLine->first_index
 *  - the current element up to which we limit the events shown : EventLine->current_index
 *  - the last element that can be plotted : EventLine->last_index
 *
 *
 * @return the #picviz_event_t object, NULL if error
 */
LibPicvizDecl picviz_eventline_t *picviz_eventline_new(PVRow row_count);
LibPicvizDecl void picviz_eventline_destroy(picviz_eventline_t *eventline);



/******************************************************************************
 ******************************************************************************
 *
 * ALL OTHER FUNCTIONS
 *
 ******************************************************************************
 *****************************************************************************/

LibPicvizDecl int picviz_eventline_get_current_index(picviz_eventline_t *eventline);
LibPicvizDecl int picviz_eventline_get_first_index(picviz_eventline_t *eventline);
LibPicvizDecl float picviz_eventline_get_kth_slider_position(picviz_eventline_t *eventline, int k);
LibPicvizDecl int picviz_eventline_get_last_index(picviz_eventline_t *eventline);

LibPicvizDecl void picviz_eventline_set_current_index(picviz_eventline_t *eventline, int index);
LibPicvizDecl void picviz_eventline_set_first_index(picviz_eventline_t *eventline, int index);
LibPicvizDecl float picviz_eventline_set_kth_index_and_adjust_slider_position(picviz_eventline_t *eventline, int k, float x);
LibPicvizDecl void picviz_eventline_set_last_index(picviz_eventline_t *eventline, int index);

LibPicvizDecl void picviz_eventline_selection_A2A_filter(picviz_eventline_t *eventline, picviz_selection_t *selection);
LibPicvizDecl void picviz_eventline_selection_A2B_filter(picviz_eventline_t *eventline, picviz_selection_t *a, picviz_selection_t *b);


#ifdef __cplusplus
 }
#endif

#endif /* _PICVIZ_EVENTLINE_H_ */
