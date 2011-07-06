//! \file eventline.cpp
//! $Id: eventline.cpp 2489 2011-04-25 01:53:05Z psaade $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <stdio.h>
#include <stdlib.h>

#include <picviz/eventline.h>
#include <picviz/selection.h>

// Some test


/******************************************************************************
 ******************************************************************************
 *
 * CREATOR/DESTRUCTOR
 *
 ******************************************************************************
 *****************************************************************************/

/******************************************************************************
 *
 * picviz_eventline_new
 *
 *****************************************************************************/
picviz_eventline_t *picviz_eventline_new(PVRow row_count)
{
	picviz_eventline_t *eventline;

	eventline = (picviz_eventline_t *)malloc(sizeof(picviz_eventline_t));
	if (!eventline) {
		fprintf(stderr, "Cannot allocate eventline!\n");
		return NULL;
	}
	
	eventline->row_count = row_count;

	eventline->first_index = 0;
	eventline->current_index = row_count - 1;
	eventline->last_index =  row_count - 1;

	return eventline;
}



/******************************************************************************
 *
 * picviz_eventline_destroy
 *
 *****************************************************************************/
void picviz_eventline_destroy(picviz_eventline_t *eventline)
{
	free(eventline);
}



/******************************************************************************
 ******************************************************************************
 *
 * ALL OTHER FUNCTIONS
 *
 ******************************************************************************
 *****************************************************************************/

/******************************************************************************
 *
 * picviz_eventline_get_current_index
 *
 *****************************************************************************/
int picviz_eventline_get_current_index(picviz_eventline_t *eventline)
{
	return eventline->current_index;
}



/******************************************************************************
 *
 * picviz_eventline_get_first_index
 *
 *****************************************************************************/
int picviz_eventline_get_first_index(picviz_eventline_t *eventline)
{
	return eventline->first_index;
}



/******************************************************************************
 *
* picviz_eventline_get_kth_slider_position
 *
 *****************************************************************************/
float picviz_eventline_get_kth_slider_position(picviz_eventline_t *eventline, int k)
{
	/* VARIABLES */
	float slider_position;

	/* CODE */
	/* We need to check that we are not in a weird situation where there would be only one row... */
	if (eventline->row_count > 1) {
		/* There is more than one row : we can compute the float */
		switch (k) {
			case 0:
				slider_position = (float)(eventline->first_index) / (eventline->row_count -1);
				break;

			case 1:
				slider_position = (float)(eventline->current_index) / (eventline->row_count -1);
				break;

			case 2:
				slider_position = (float)(eventline->last_index) / (eventline->row_count -1);
				break;
		}

		return slider_position;
	} else {
		/* There is only one row ! */
		return (float)0.0;
	}
}



/******************************************************************************
 *
 * picviz_eventline_get_last_index
 *
 *****************************************************************************/
int picviz_eventline_get_last_index(picviz_eventline_t *eventline)
{
	return eventline->last_index;
}



/******************************************************************************
 *
 * picviz_eventline_selection_A2A_filter
 *
 *****************************************************************************/
void picviz_eventline_selection_A2A_filter(picviz_eventline_t *eventline, picviz_selection_t *selection)
{
		int i;
		/* We unselect all lines before first_index */
		for (i=0; i<eventline->first_index; i++) {
			picviz_selection_set_line(selection, i, 0);
		}
		/* We unselect all lines after current_index */
		for (i=eventline->current_index + 1; i<eventline->row_count; i++) {
			picviz_selection_set_line(selection, i, 0);
		}
}



/******************************************************************************
 *
 * picviz_eventline_selection_A2B_filter
 *
 *****************************************************************************/
void picviz_eventline_selection_A2B_filter(picviz_eventline_t *eventline, picviz_selection_t *a, picviz_selection_t *b)
{
	picviz_selection_A2B_copy(a,b);
	picviz_eventline_selection_A2A_filter(eventline, b);
}



/******************************************************************************
 *
 * picviz_eventline_set_current_index
 *
 *****************************************************************************/
void picviz_eventline_set_current_index(picviz_eventline_t *eventline, int index)
{
	eventline->current_index = index;
}



/******************************************************************************
 *
 * picviz_eventline_set_first_index
 *
 *****************************************************************************/
void picviz_eventline_set_first_index(picviz_eventline_t *eventline, int index)
{
	eventline->first_index = index;
}



/******************************************************************************
 *
 * picviz_eventline_set_kth_index_from_float
 *
 *****************************************************************************/
float picviz_eventline_set_kth_index_and_adjust_slider_position(picviz_eventline_t *eventline, int k, float x)
{
	/* VARIABLES */
	int real_index;
	float real_position;
	
	/* CODE */
	/* We compute the int position closest the x*row_count */
	real_index = (int)(x*(eventline->row_count - 1) + 0.5);
	
	/* depending on k, we change the index */
	switch (k) {
		case 0:
			if (real_index < 0) {
				real_index = 0;
			}
			if (real_index > eventline->current_index) {
				real_index = eventline->current_index;
			}
			eventline->first_index = real_index;
			break;

		case 1:
			if (real_index < eventline->first_index) {
				real_index = eventline->first_index;
			}
			if (real_index > eventline->last_index) {
				real_index = eventline->last_index;
			}
			eventline->current_index = real_index;
			break;

		case 2:
			if (real_index < eventline->current_index) {
				real_index = eventline->current_index;
			}
			if (real_index >= eventline->row_count) {
				real_index = eventline->row_count -1;
			}
			eventline->last_index = real_index;
			break;
	}

	/* Now we adjust the float position of the concerned slider */
	if (eventline->row_count > 1) {
		real_position = (float)(real_index) / (eventline->row_count - 1);
		return real_position;
	} else {
		return (float)0.0;
	}
}



/******************************************************************************
 *
 * picviz_eventline_set_last_index
 *
 *****************************************************************************/
void picviz_eventline_set_last_index(picviz_eventline_t *eventline, int index)
{
	eventline->last_index = index;
}


