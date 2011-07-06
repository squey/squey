//! \file filter-unit.cpp
//! $Id: filter-unit.cpp 2489 2011-04-25 01:53:05Z psaade $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <stdio.h>
#include <stdlib.h>

#include <apr_general.h>

#include <picviz/filter-unit.h>



/******************************************************************************
 ******************************************************************************
 *
 * CREATOR/DESTRUCTOR
 *
 ******************************************************************************
 *****************************************************************************/

/******************************************************************************
 *
 * picviz_filter_unit_new
 *
 *****************************************************************************/
picviz_filter_unit_t *picviz_filter_unit_new(void)
{
	picviz_filter_unit_t *filter_unit;

	filter_unit = (picviz_filter_unit_t *)malloc(sizeof(picviz_filter_unit_t));
	filter_unit->arguments_count = 0;
	return filter_unit;
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
 * picviz_filter_unit_argument_append
 *
 *****************************************************************************/
void picviz_filter_unit_argument_append(picviz_filter_unit_t *filter_unit, picviz_filter_unit_argument_t argument)
{
	if (filter_unit->arguments_count >= PICVIZ_FILTER_UNIT_MAXARGS) {
		fprintf(stderr, "Error(%s): cannot append a new argument!\n", __FUNCTION__);
		return;
	}
	filter_unit->arguments[filter_unit->arguments_count] = argument;
	filter_unit->arguments_count++;
}



/******************************************************************************
 *
 * picviz_filter_unit_argument_append_char_p
 *
 *****************************************************************************/
void picviz_filter_unit_argument_append_char_p(picviz_filter_unit_t *filter_unit, char *arg)
{
	picviz_filter_unit_argument_t argument;
	argument.a_char_p = arg;
	picviz_filter_unit_argument_append(filter_unit, argument);
}



/******************************************************************************
 *
 * picviz_filter_unit_argument_append_float
 *
 *****************************************************************************/
void picviz_filter_unit_argument_append_float(picviz_filter_unit_t *filter_unit, float arg)
{
	picviz_filter_unit_argument_t argument;
	argument.a_float = arg;
	picviz_filter_unit_argument_append(filter_unit, argument);
}



/******************************************************************************
 *
 * picviz_filter_unit_argument_get_char
 *
 *****************************************************************************/
char picviz_filter_unit_argument_get_char(picviz_filter_unit_t *filter_unit, int position)
{
	return filter_unit->arguments[position].a_char;
}



/******************************************************************************
 *
 * picviz_filter_unit_argument_get_char_p
 *
 *****************************************************************************/
char *picviz_filter_unit_argument_get_char_p(picviz_filter_unit_t *filter_unit, int position)
{
	return filter_unit->arguments[position].a_char_p;
}



/******************************************************************************
 *
 * picviz_filter_unit_argument_get_float
 *
 *****************************************************************************/
float picviz_filter_unit_argument_get_float(picviz_filter_unit_t *filter_unit, int position)
{
	return filter_unit->arguments[position].a_float;
}



/******************************************************************************
 *
 * picviz_filter_unit_argument_get_int
 *
 *****************************************************************************/
int picviz_filter_unit_argument_get_int(picviz_filter_unit_t *filter_unit, int position)
{
	return filter_unit->arguments[position].a_int;
}



/******************************************************************************
 *
 * picviz_filter_unit_argument_get_pointer
 *
 *****************************************************************************/
void *picviz_filter_unit_argument_get_pointer(picviz_filter_unit_t *filter_unit, int position)
{
	return filter_unit->arguments[position].a_pointer;
}



/******************************************************************************
 *
 * picviz_filter_unit_argument_get_uint64
 *
 *****************************************************************************/
apr_uint64_t picviz_filter_unit_argument_get_uint64(picviz_filter_unit_t *filter_unit, int position)
{
	return filter_unit->arguments[position].a_uint64;
}






/* picviz_filter_unit_ */
#ifdef _UNIT_TEST_
int main(void)
{
	picviz_filter_unit_t *filter_unit;
	picviz_filter_unit_argument_t argument;
	picviz_filter_unit_argument_t arg_char_p;

	filter_unit = picviz_filter_unit_new();

	picviz_filter_unit_argument_append_float(filter_unit, 0.123);
	picviz_filter_unit_argument_append_char_p(filter_unit, "coucou");

	printf("float for position 0 = %f\n", picviz_filter_unit_argument_get_float(filter_unit, 0));
	printf("string for position 1 = %s\n", picviz_filter_unit_argument_get_char_p(filter_unit, 1));

	free(filter_unit);

	return 0;
}
#endif
