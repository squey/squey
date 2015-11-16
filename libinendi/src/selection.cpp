/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <inendi/general.h>
#include <inendi/selection.h>

#include "bithacks.h"

unsigned int line_count = 0;

/******************************************************************************
 ******************************************************************************
 *
 * CREATOR/DESTRUCTOR
 *
 ******************************************************************************
 *****************************************************************************/


/******************************************************************************
 *
 * inendi_selection_new
 *
 *****************************************************************************/
inendi_selection_t *inendi_selection_new(void)
{
	inendi_selection_t *ts;

	ts = (inendi_selection_t *)malloc(sizeof(inendi_selection_t));
	ts->table = (uint32_t *)calloc(INENDI_SELECTION_NUMBER_OF_CHUNKS, sizeof(uint32_t));
	if ( ! ts->table ) {
		fprintf(stderr, "Cannot allocate table selection!\n");
		return NULL;
	}
	memset(ts->table, 0x00, INENDI_SELECTION_NUMBER_OF_BYTES);

	return ts;
}



/******************************************************************************
 *
 * inendi_selection_destroy
 *
 *****************************************************************************/
void inendi_selection_destroy(inendi_selection_t *selection)
{
	free(selection->table);
	free(selection);
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
 * inendi_selection_get_line
 *
 *****************************************************************************/
int inendi_selection_get_line(inendi_selection_t *selection, pv_row line_id)
{
	pv_row pos;
	pv_row shift;

	/*
	 * Say you want to retrieve if the line 20000 is selected or not:
	 * pos = 312
	 * shift = 32
	 */
	pos = line_id / INENDI_SELECTION_CHUNK_SIZE;
	shift = line_id - (pos * INENDI_SELECTION_CHUNK_SIZE);

	return B_IS_SET(selection->table[pos], shift);

}


/******************************************************************************
 *
 * inendi_selection_get_line_index_of_nth_selected_line
 *
 *****************************************************************************/
/* give 1 for index: selection = 00100: return 3 */
// pv_row inendi_selection_get_line_index_of_nth_selected_line(inendi_selection_t *selection, pv_row n)
// {
// 	pv_row line_index;
// 	pv_row count;
// 
// 	count = 0;
// 	for (line_index = 0; line_index < INENDI_LINES_MAX; line_index++) {
// 		if (inendi_selection_get_line(selection, line_index)) {
// 			count++;
// 			if (count == n) {
// 				return line_index;
// 			}
// 		}
// 	}
// 
// 	return 0;
// }



/******************************************************************************
 *
 * inendi_selection_get_number_of_selected_lines_in_range
 * WARNING !
 * line a BELONGS to the range but line b DOES NOT !!!
 *
 *****************************************************************************/
int inendi_selection_get_number_of_selected_lines_in_range(inendi_selection_t *selection, pv_row a, pv_row b)
{
	pv_row line_index;
	int count = 0;

	for (line_index = a; line_index<b; line_index++) {
		if (inendi_selection_get_line(selection, line_index)) {
			count++;
		}
	}

	return count;
}



/******************************************************************************
 *
 * inendi_selection_set_line
 *
 *****************************************************************************/
void inendi_selection_set_line(inendi_selection_t *ts, pv_row lineid, int bool_value)
{
	pv_row pos;
	pv_row shift;

	pos = lineid / INENDI_SELECTION_CHUNK_SIZE;
	shift = lineid - (pos * INENDI_SELECTION_CHUNK_SIZE);

	if ( bool_value )  {
		B_SET(ts->table[pos], shift);
	} else {
		B_UNSET(ts->table[pos], shift);
	}
}





/******************************************************************************
 ******************************************************************************
 *
 * Generic functions (that do not act as Operators on selections).
 *
 *****************************************************************************
 *****************************************************************************/




static int _inendi_selection_printf(long offset, char *line, size_t linesize, void *userdata)
{
	inendi_selection_t *selection = (inendi_selection_t *)userdata;
	int line_selected;
	
	line_selected = inendi_selection_get_line(selection, line_count);
	if (line_selected) {
		printf("%s\n", line);
	}

	line_count++;
}


/******************************************************************************
 ******************************************************************************
 *
 * the set of operators of type A2A : inplace on A
 *
 ******************************************************************************
 *****************************************************************************/

/******************************************************************************
 *
 * inendi_selection_A2A_inverse
 *
 *****************************************************************************/
void inendi_selection_A2A_inverse(inendi_selection_t *a)
{
	pv_row i;

	for (i=0; i < INENDI_SELECTION_NUMBER_OF_CHUNKS; i++) {
		a->table[i] =~ a->table[i];
	}
}

/******************************************************************************
 *
 * inendi_selection_A2A_select_all
 *
 *****************************************************************************/
void inendi_selection_A2A_select_all(inendi_selection_t *a)
{
	memset(a->table, 0xFF, INENDI_SELECTION_NUMBER_OF_BYTES);
}


/******************************************************************************
 *
 * inendi_selection_A2A_select_even
 *
 *****************************************************************************/
void inendi_selection_A2A_select_even(inendi_selection_t *a)
{
	// THE VALUE SHOULD BE 0xAA !!!
	memset(a->table, 0x55, INENDI_SELECTION_NUMBER_OF_BYTES);
}


/******************************************************************************
 *
 * inendi_selection_A2A_select_even
 *
 *****************************************************************************/
void inendi_selection_A2A_select_from_s_to_e(inendi_selection_t *a, int start, int end)
{

}


/******************************************************************************
 *
 * inendi_selection_A2A_select_none
 *
 *****************************************************************************/
void inendi_selection_A2A_select_none(inendi_selection_t *a)
{
	memset(a->table, 0x00, INENDI_SELECTION_NUMBER_OF_BYTES);
}


/******************************************************************************
 *
 * inendi_selection_A2A_select_odd
 *
 *****************************************************************************/
void inendi_selection_A2A_select_odd(inendi_selection_t *a)
{
	// THE VALUE SHOULD BE 0x55!!!!
	memset(a->table, 0xAA, INENDI_SELECTION_NUMBER_OF_BYTES);
}




/******************************************************************************
 ******************************************************************************
 *
 * the set of operators of type A2B : Operator(A, B) : A --> B
 *
 ******************************************************************************
 *****************************************************************************/

/******************************************************************************
 *
 * inendi_selection_A2B_copy
 *
 *****************************************************************************/
void inendi_selection_A2B_copy(inendi_selection_t *a, inendi_selection_t *b)
{
	memcpy(b->table, a->table, INENDI_SELECTION_NUMBER_OF_BYTES);
}


/******************************************************************************
 *
 * inendi_selection_A2B_inverse
 *
 *****************************************************************************/
void inendi_selection_A2B_inverse(inendi_selection_t *a, inendi_selection_t *b)
{
	pv_row i;

	for (i=0; i < INENDI_SELECTION_NUMBER_OF_CHUNKS; i++) {
		b->table[i] =~ a->table[i];
	}
}





/******************************************************************************
 ******************************************************************************
 *
 * the set of operators of type AB2A : (A operator B) --> A
 *
 ******************************************************************************
 *****************************************************************************/

/******************************************************************************
 *
 * inendi_selection_AB2A_and
 *
 *****************************************************************************/
void inendi_selection_AB2A_and(inendi_selection_t *a, inendi_selection_t *b)
{
	pv_row i;

	for (i=0; i < INENDI_SELECTION_NUMBER_OF_CHUNKS; i++) {
		a->table[i] = a->table[i] & b->table[i];
	}
}


/******************************************************************************
 *
 * inendi_selection_AB2A_or
 *
 *****************************************************************************/
void inendi_selection_AB2A_or(inendi_selection_t *a, inendi_selection_t *b)
{
	pv_row i;

	for (i=0; i < INENDI_SELECTION_NUMBER_OF_CHUNKS; i++) {
		a->table[i] = a->table[i] | b->table[i];
	}
}


/******************************************************************************
 *
 * inendi_selection_AB2A_substraction
 *
 *****************************************************************************/
void inendi_selection_AB2A_substraction(inendi_selection_t *a, inendi_selection_t *b)
{
	pv_row i;

	for (i=0; i < INENDI_SELECTION_NUMBER_OF_CHUNKS; i++) {
		a->table[i] = a->table[i] & ~(b->table[i]);
	}
}


/******************************************************************************
 *
 * inendi_selection_AB2A_xor
 *
 *****************************************************************************/
void inendi_selection_AB2A_xor(inendi_selection_t *a, inendi_selection_t *b)
{
	pv_row i;

	for (i=0; i < INENDI_SELECTION_NUMBER_OF_CHUNKS; i++) {
		a->table[i] = a->table[i] ^ b->table[i];
	}
}




/******************************************************************************
 ******************************************************************************
 *
 * the set of operators of type AB2C : (A operator B) --> C
 *
 ******************************************************************************
 *****************************************************************************/

/******************************************************************************
 *
 * inendi_selection_AB2C_and
 *
 *****************************************************************************/
void inendi_selection_AB2C_and(inendi_selection_t *a, inendi_selection_t *b, inendi_selection_t *c)
{
	pv_row i;

	for (i=0; i < INENDI_SELECTION_NUMBER_OF_CHUNKS; i++) {
		c->table[i] = a->table[i] & b->table[i];
	}
}


/******************************************************************************
 *
 * inendi_selection_AB2C_or
 *
 *****************************************************************************/
void inendi_selection_AB2C_or(inendi_selection_t *a, inendi_selection_t *b, inendi_selection_t *c)
{
	pv_row i;

	for (i=0; i < INENDI_SELECTION_NUMBER_OF_CHUNKS; i++) {
		c->table[i] = a->table[i] | b->table[i];
	}
}


/******************************************************************************
 *
 * inendi_selection_AB2C_substraction
 *
 *****************************************************************************/
void inendi_selection_AB2C_substraction(inendi_selection_t *a, inendi_selection_t *b, inendi_selection_t *c)
{
	pv_row i;

	for (i=0; i < INENDI_SELECTION_NUMBER_OF_CHUNKS; i++) {
		c->table[i] = a->table[i] & ~(b->table[i]);
	}
}


/******************************************************************************
 *
 * inendi_selection_AB2C_xor
 *
 *****************************************************************************/
void inendi_selection_AB2C_xor(inendi_selection_t *a, inendi_selection_t *b, inendi_selection_t *c)
{
	pv_row i;

	for (i=0; i < INENDI_SELECTION_NUMBER_OF_CHUNKS; i++) {
		c->table[i] = a->table[i] ^ b->table[i];
	}
}







/******************************************************************************
 ******************************************************************************
 *
 * Old Unit Test
 *
 ******************************************************************************
 *****************************************************************************/


#ifdef _UNIT_TEST_
int test_selection(void)
{
	inendi_selection_t *selection;

	selection = inendi_selection_new();
	printf("Line 12345 selection status: '%d'\n", inendi_selection_get_line(selection, 12345));
	inendi_selection_set_line(selection, 12345, 1);
	printf("Line 12345 selection status: '%d'\n", inendi_selection_get_line(selection, 12345));

	return 0;
}

int main(void)
{
	return test_selection();
}
#endif
