/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <stdio.h>
#include <stdlib.h>

#include <apr_general.h>

#include <inendi/filter-unit.h>



/******************************************************************************
 ******************************************************************************
 *
 * CREATOR/DESTRUCTOR
 *
 ******************************************************************************
 *****************************************************************************/

/******************************************************************************
 *
 * inendi_filter_unit_new
 *
 *****************************************************************************/
inendi_filter_unit_t *inendi_filter_unit_new(void)
{
	inendi_filter_unit_t *filter_unit;

	filter_unit = (inendi_filter_unit_t *)malloc(sizeof(inendi_filter_unit_t));
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
 * inendi_filter_unit_argument_append
 *
 *****************************************************************************/
void inendi_filter_unit_argument_append(inendi_filter_unit_t *filter_unit, inendi_filter_unit_argument_t argument)
{
	if (filter_unit->arguments_count >= INENDI_FILTER_UNIT_MAXARGS) {
		fprintf(stderr, "Error(%s): cannot append a new argument!\n", __FUNCTION__);
		return;
	}
	filter_unit->arguments[filter_unit->arguments_count] = argument;
	filter_unit->arguments_count++;
}



/******************************************************************************
 *
 * inendi_filter_unit_argument_append_char_p
 *
 *****************************************************************************/
void inendi_filter_unit_argument_append_char_p(inendi_filter_unit_t *filter_unit, char *arg)
{
	inendi_filter_unit_argument_t argument;
	argument.a_char_p = arg;
	inendi_filter_unit_argument_append(filter_unit, argument);
}



/******************************************************************************
 *
 * inendi_filter_unit_argument_append_float
 *
 *****************************************************************************/
void inendi_filter_unit_argument_append_float(inendi_filter_unit_t *filter_unit, float arg)
{
	inendi_filter_unit_argument_t argument;
	argument.a_float = arg;
	inendi_filter_unit_argument_append(filter_unit, argument);
}



/******************************************************************************
 *
 * inendi_filter_unit_argument_get_char
 *
 *****************************************************************************/
char inendi_filter_unit_argument_get_char(inendi_filter_unit_t *filter_unit, int position)
{
	return filter_unit->arguments[position].a_char;
}



/******************************************************************************
 *
 * inendi_filter_unit_argument_get_char_p
 *
 *****************************************************************************/
char *inendi_filter_unit_argument_get_char_p(inendi_filter_unit_t *filter_unit, int position)
{
	return filter_unit->arguments[position].a_char_p;
}



/******************************************************************************
 *
 * inendi_filter_unit_argument_get_float
 *
 *****************************************************************************/
float inendi_filter_unit_argument_get_float(inendi_filter_unit_t *filter_unit, int position)
{
	return filter_unit->arguments[position].a_float;
}



/******************************************************************************
 *
 * inendi_filter_unit_argument_get_int
 *
 *****************************************************************************/
int inendi_filter_unit_argument_get_int(inendi_filter_unit_t *filter_unit, int position)
{
	return filter_unit->arguments[position].a_int;
}



/******************************************************************************
 *
 * inendi_filter_unit_argument_get_pointer
 *
 *****************************************************************************/
void *inendi_filter_unit_argument_get_pointer(inendi_filter_unit_t *filter_unit, int position)
{
	return filter_unit->arguments[position].a_pointer;
}



/******************************************************************************
 *
 * inendi_filter_unit_argument_get_uint64
 *
 *****************************************************************************/
apr_uint64_t inendi_filter_unit_argument_get_uint64(inendi_filter_unit_t *filter_unit, int position)
{
	return filter_unit->arguments[position].a_uint64;
}






/* inendi_filter_unit_ */
#ifdef _UNIT_TEST_
int main(void)
{
	inendi_filter_unit_t *filter_unit;
	inendi_filter_unit_argument_t argument;
	inendi_filter_unit_argument_t arg_char_p;

	filter_unit = inendi_filter_unit_new();

	inendi_filter_unit_argument_append_float(filter_unit, 0.123);
	inendi_filter_unit_argument_append_char_p(filter_unit, "coucou");

	printf("float for position 0 = %f\n", inendi_filter_unit_argument_get_float(filter_unit, 0));
	printf("string for position 1 = %s\n", inendi_filter_unit_argument_get_char_p(filter_unit, 1));

	free(filter_unit);

	return 0;
}
#endif
