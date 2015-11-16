/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <stdio.h>
#include <stdlib.h>

#include <inendi/eventline.h>
#include <inendi/selection.h>

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
 * inendi_eventline_new
 *
 *****************************************************************************/
inendi_eventline_t *inendi_eventline_new(PVRow row_count)
{
	inendi_eventline_t *eventline;

	eventline = (inendi_eventline_t *)malloc(sizeof(inendi_eventline_t));
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
 * inendi_eventline_destroy
 *
 *****************************************************************************/
void inendi_eventline_destroy(inendi_eventline_t *eventline)
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
 * inendi_eventline_get_current_index
 *
 *****************************************************************************/
int inendi_eventline_get_current_index(inendi_eventline_t *eventline)
{
	return eventline->current_index;
}



/******************************************************************************
 *
 * inendi_eventline_get_first_index
 *
 *****************************************************************************/
int inendi_eventline_get_first_index(inendi_eventline_t *eventline)
{
	return eventline->first_index;
}



/******************************************************************************
 *
* inendi_eventline_get_kth_slider_position
 *
 *****************************************************************************/
float inendi_eventline_get_kth_slider_position(inendi_eventline_t *eventline, int k)
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
 * inendi_eventline_get_last_index
 *
 *****************************************************************************/
int inendi_eventline_get_last_index(inendi_eventline_t *eventline)
{
	return eventline->last_index;
}



/******************************************************************************
 *
 * inendi_eventline_selection_A2A_filter
 *
 *****************************************************************************/
void inendi_eventline_selection_A2A_filter(inendi_eventline_t *eventline, inendi_selection_t *selection)
{
		int i;
		/* We unselect all lines before first_index */
		for (i=0; i<eventline->first_index; i++) {
			inendi_selection_set_line(selection, i, 0);
		}
		/* We unselect all lines after current_index */
		for (i=eventline->current_index + 1; i<eventline->row_count; i++) {
			inendi_selection_set_line(selection, i, 0);
		}
}



/******************************************************************************
 *
 * inendi_eventline_selection_A2B_filter
 *
 *****************************************************************************/
void inendi_eventline_selection_A2B_filter(inendi_eventline_t *eventline, inendi_selection_t *a, inendi_selection_t *b)
{
	inendi_selection_A2B_copy(a,b);
	inendi_eventline_selection_A2A_filter(eventline, b);
}



/******************************************************************************
 *
 * inendi_eventline_set_current_index
 *
 *****************************************************************************/
void inendi_eventline_set_current_index(inendi_eventline_t *eventline, int index)
{
	eventline->current_index = index;
}



/******************************************************************************
 *
 * inendi_eventline_set_first_index
 *
 *****************************************************************************/
void inendi_eventline_set_first_index(inendi_eventline_t *eventline, int index)
{
	eventline->first_index = index;
}



/******************************************************************************
 *
 * inendi_eventline_set_kth_index_from_float
 *
 *****************************************************************************/
float inendi_eventline_set_kth_index_and_adjust_slider_position(inendi_eventline_t *eventline, int k, float x)
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
 * inendi_eventline_set_last_index
 *
 *****************************************************************************/
void inendi_eventline_set_last_index(inendi_eventline_t *eventline, int index)
{
	eventline->last_index = index;
}


