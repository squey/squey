//! \file square-area.cpp
//! $Id: square-area.cpp 2489 2011-04-25 01:53:05Z psaade $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <pvcore/general.h>

#include <picviz/general.h>
#include <picviz/square-area.h>


/******************************************************************************
 ******************************************************************************
 *
 * CREATOR/DESTRUCTOR
 *
 ******************************************************************************
 *****************************************************************************/


/******************************************************************************
 *
 * picviz_square_area_new
 *
 *****************************************************************************/
picviz_square_area_t *picviz_square_area_new(void)
{
	picviz_square_area_t *sa;

	sa = (picviz_square_area_t *)malloc(sizeof(picviz_square_area_t));
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
 * picviz_square_area_destroy
 *
 *****************************************************************************/
void picviz_square_area_destroy(picviz_square_area_t *sa)
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
 * picviz_square_area_get_end_x
 *
 *****************************************************************************/
float picviz_square_area_get_end_x(picviz_square_area_t *sa)
{
	return sa->end_x;
}



/******************************************************************************
 *
 * picviz_square_area_get_end_y
 *
 *****************************************************************************/
float picviz_square_area_get_end_y(picviz_square_area_t *sa)
{
	return sa->end_y;
}



/******************************************************************************
 *
 * picviz_square_area_get_start_x
 *
 *****************************************************************************/
float picviz_square_area_get_start_x(picviz_square_area_t *sa)
{
	return sa->start_x;
}



/******************************************************************************
 *
 * picviz_square_area_get_start_y
 *
 *****************************************************************************/
float picviz_square_area_get_start_y(picviz_square_area_t *sa)
{
	return sa->start_y;
}



/******************************************************************************
 *
 * picviz_square_area_set_end
 *
 *****************************************************************************/
void picviz_square_area_set_end(picviz_square_area_t *sa, float ex, float ey)
{
	sa->end_x = ex;
	sa->end_y = ey;
}



/******************************************************************************
 *
 * picviz_square_area_set_end_x
 *
 *****************************************************************************/
void picviz_square_area_set_end_x(picviz_square_area_t *sa, float ex)
{
	sa->end_x = ex;
}



/******************************************************************************
 *
 * picviz_square_area_set_end_y
 *
 *****************************************************************************/
void picviz_square_area_set_end_y(picviz_square_area_t *sa, float ey)
{
	sa->end_y = ey;
}



/******************************************************************************
 *
 * picviz_square_area_set_start
 *
 *****************************************************************************/
void picviz_square_area_set_start(picviz_square_area_t *sa, float sx, float sy)
{
	sa->start_x = sx;
	sa->start_y = sy;
}



/******************************************************************************
 *
 * picviz_square_area_set_start_x
 *
 *****************************************************************************/
void picviz_square_area_set_start_x(picviz_square_area_t *sa, float sx)
{
	sa->start_x = sx;
}

/******************************************************************************
 *
 * picviz_square_area_
 *
 *****************************************************************************/
void picviz_square_area_set_start_y(picviz_square_area_t *sa, float sy)
{
	sa->start_y = sy;
}


