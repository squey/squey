/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <pvkernel/core/general.h>

#include <inendi/general.h>
#include <inendi/square-area.h>


/******************************************************************************
 ******************************************************************************
 *
 * CREATOR/DESTRUCTOR
 *
 ******************************************************************************
 *****************************************************************************/


/******************************************************************************
 *
 * inendi_square_area_new
 *
 *****************************************************************************/
inendi_square_area_t *inendi_square_area_new(void)
{
	inendi_square_area_t *sa;

	sa = (inendi_square_area_t *)malloc(sizeof(inendi_square_area_t));
	if (!sa) {
		PVLOG_ERROR("Cannot allocate square-area in %s!\n", __FUNCTION__);
		return NULL;
	}

	sa->end_x = 0.0;
	sa->end_y = 0.0;
	sa->start_x = 0.0;
	sa->start_y = 0.0;
	
	return sa;
}



/******************************************************************************
 *
 * inendi_square_area_destroy
 *
 *****************************************************************************/
void inendi_square_area_destroy(inendi_square_area_t *sa)
{
	free(sa);
}



/******************************************************************************
 ******************************************************************************
 *
 * GET/SET functions
 *
 ******************************************************************************
 *****************************************************************************/

/******************************************************************************
 *
 * inendi_square_area_get_end_x
 *
 *****************************************************************************/
float inendi_square_area_get_end_x(inendi_square_area_t *sa)
{
	return sa->end_x;
}



/******************************************************************************
 *
 * inendi_square_area_get_end_y
 *
 *****************************************************************************/
float inendi_square_area_get_end_y(inendi_square_area_t *sa)
{
	return sa->end_y;
}



/******************************************************************************
 *
 * inendi_square_area_get_start_x
 *
 *****************************************************************************/
float inendi_square_area_get_start_x(inendi_square_area_t *sa)
{
	return sa->start_x;
}



/******************************************************************************
 *
 * inendi_square_area_get_start_y
 *
 *****************************************************************************/
float inendi_square_area_get_start_y(inendi_square_area_t *sa)
{
	return sa->start_y;
}



/******************************************************************************
 *
 * inendi_square_area_set_end
 *
 *****************************************************************************/
void inendi_square_area_set_end(inendi_square_area_t *sa, float ex, float ey)
{
	sa->end_x = ex;
	sa->end_y = ey;
}



/******************************************************************************
 *
 * inendi_square_area_set_end_x
 *
 *****************************************************************************/
void inendi_square_area_set_end_x(inendi_square_area_t *sa, float ex)
{
	sa->end_x = ex;
}



/******************************************************************************
 *
 * inendi_square_area_set_end_y
 *
 *****************************************************************************/
void inendi_square_area_set_end_y(inendi_square_area_t *sa, float ey)
{
	sa->end_y = ey;
}



/******************************************************************************
 *
 * inendi_square_area_set_start
 *
 *****************************************************************************/
void inendi_square_area_set_start(inendi_square_area_t *sa, float sx, float sy)
{
	sa->start_x = sx;
	sa->start_y = sy;
}



/******************************************************************************
 *
 * inendi_square_area_set_start_x
 *
 *****************************************************************************/
void inendi_square_area_set_start_x(inendi_square_area_t *sa, float sx)
{
	sa->start_x = sx;
}

/******************************************************************************
 *
 * inendi_square_area_
 *
 *****************************************************************************/
void inendi_square_area_set_start_y(inendi_square_area_t *sa, float sy)
{
	sa->start_y = sy;
}


