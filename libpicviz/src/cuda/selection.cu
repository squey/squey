/*
 * $Id: selection.cpp 2017 2011-02-28 19:10:49Z stricaud $
 * Copyright (C) Sebastien Tricaud 2010
 * Copyright (C) Philippe Saade 2010
 * 
 */

#include <stdio.h>
#include <stdlib.h>

#include <picviz/general.h>
#include <picviz/selection.h>

#include <bithacks.h>

#include <cuda.h>
//#include <cutil_inline.h>

#include "selection_kernels.cu"


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
 * picviz_selection_new
 *
 *****************************************************************************/
picviz_selection_t *picviz_selection_new(void)
{
	picviz_selection_t *ts;

	ts = (picviz_selection_t *)malloc(sizeof(picviz_selection_t));
	ts->table = (uint32_t *)calloc(PICVIZ_SELECTION_NUMBER_OF_CHUNKS, sizeof(uint32_t));
	if ( ! ts->table ) {
		fprintf(stderr, "Cannot allocate table selection!\n");
		return NULL;
	}
	memset(ts->table, 0x00, PICVIZ_SELECTION_NUMBER_OF_BYTES);

	return ts;
}



/******************************************************************************
 *
 * picviz_selection_destroy
 *
 *****************************************************************************/
void picviz_selection_destroy(picviz_selection_t *selection)
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
 * picviz_selection_get_line
 *
 *****************************************************************************/
int picviz_selection_get_line(picviz_selection_t *selection, pv_row line_id)
{
	pv_row pos;
	pv_row shift;

	/*
	 * Say you want to retrieve if the line 20000 is selected or not:
	 * pos = 312
	 * shift = 32
	 */
	pos = line_id / PICVIZ_SELECTION_CHUNK_SIZE;
	shift = line_id - (pos * PICVIZ_SELECTION_CHUNK_SIZE);

	return B_IS_SET(selection->table[pos], shift);

}


/******************************************************************************
 *
 * picviz_selection_get_line_index_of_nth_selected_line
 *
 *****************************************************************************/
/* give 1 for index: selection = 00100: return 3 */
pv_row picviz_selection_get_line_index_of_nth_selected_line(picviz_selection_t *selection, pv_row n)
{
	pv_row line_index;
	pv_row count;

	count = 0;
	for (line_index = 0; line_index < PICVIZ_LINES_MAX; line_index++) {
		if (picviz_selection_get_line(selection, line_index)) {
			count++;
			if (count == n) {
				return line_index;
			}
		}
	}

	return 0;
}



/******************************************************************************
 *
 * picviz_selection_get_number_of_selected_lines_in_range
 * WARNING !
 * line a BELONGS to the range but line b DOES NOT !!!
 *
 *****************************************************************************/
int picviz_selection_get_number_of_selected_lines_in_range(picviz_selection_t *selection, pv_row a, pv_row b)
{
	pv_row line_index;
	int count = 0;

	for (line_index = a; line_index<b; line_index++) {
		if (picviz_selection_get_line(selection, line_index)) {
			count++;
		}
	}

	return count;
}



/******************************************************************************
 *
 * picviz_selection_set_line
 *
 *****************************************************************************/
void picviz_selection_set_line(picviz_selection_t *ts, pv_row lineid, int bool_value)
{
	pv_row pos;
	pv_row shift;

	pos = lineid / PICVIZ_SELECTION_CHUNK_SIZE;
	shift = lineid - (pos * PICVIZ_SELECTION_CHUNK_SIZE);

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


/******************************************************************************
 *
 * picviz_selection_count
 *
 *****************************************************************************/
/* We need to provide lines_counts because select all will select even lines that do not exists! */
pv_row picviz_selection_count(picviz_selection_t *selection, pv_row last_line_index)
{
	pv_row index;
	pv_row count;

	count = 0;
	for (index = 0; index <= last_line_index; index++) {
		count += picviz_selection_get_line(selection, index);
	}

	return count;
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
 * picviz_selection_A2A_inverse
 *
 *****************************************************************************/
void picviz_selection_A2A_inverse(picviz_selection_t *a)
{
	pv_row i;

	for (i=0; i < PICVIZ_SELECTION_NUMBER_OF_CHUNKS; i++) {
		a->table[i] =~ a->table[i];
	}
}

/******************************************************************************
 *
 * picviz_selection_A2A_select_all
 *
 *****************************************************************************/
void picviz_selection_A2A_select_all(picviz_selection_t *a)
{
	memset(a->table, 0xFF, PICVIZ_SELECTION_NUMBER_OF_BYTES);
}


/******************************************************************************
 *
 * picviz_selection_A2A_select_even
 *
 *****************************************************************************/
void picviz_selection_A2A_select_even(picviz_selection_t *a)
{
	// THE VALUE SHOULD BE 0xAA !!!
	memset(a->table, 0x55, PICVIZ_SELECTION_NUMBER_OF_BYTES);
}


/******************************************************************************
 *
 * picviz_selection_A2A_select_even
 *
 *****************************************************************************/
void picviz_selection_A2A_select_from_s_to_e(picviz_selection_t *a, int start, int end)
{

}


/******************************************************************************
 *
 * picviz_selection_A2A_select_none
 *
 *****************************************************************************/
void picviz_selection_A2A_select_none(picviz_selection_t *a)
{
	memset(a->table, 0x00, PICVIZ_SELECTION_NUMBER_OF_BYTES);
}


/******************************************************************************
 *
 * picviz_selection_A2A_select_odd
 *
 *****************************************************************************/
void picviz_selection_A2A_select_odd(picviz_selection_t *a)
{
	// THE VALUE SHOULD BE 0x55!!!!
	memset(a->table, 0xAA, PICVIZ_SELECTION_NUMBER_OF_BYTES);
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
 * picviz_selection_A2B_copy
 *
 *****************************************************************************/
void picviz_selection_A2B_copy(picviz_selection_t *a, picviz_selection_t *b)
{
	memcpy(b->table, a->table, PICVIZ_SELECTION_NUMBER_OF_BYTES);
}


/******************************************************************************
 *
 * picviz_selection_A2B_inverse
 *
 *****************************************************************************/
void picviz_selection_A2B_inverse(picviz_selection_t *a, picviz_selection_t *b)
{
	pv_row i;

	for (i=0; i < PICVIZ_SELECTION_NUMBER_OF_CHUNKS; i++) {
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
 * picviz_selection_AB2A_and
 *
 *****************************************************************************/
void picviz_selection_AB2A_and(picviz_selection_t *a, picviz_selection_t *b)
{
	pv_row i;

	for (i=0; i < PICVIZ_SELECTION_NUMBER_OF_CHUNKS; i++) {
		a->table[i] = a->table[i] & b->table[i];
	}
}


/******************************************************************************
 *
 * picviz_selection_AB2A_or
 *
 *****************************************************************************/
void picviz_selection_AB2A_or(picviz_selection_t *a, picviz_selection_t *b)
{
	void *da;
	void *db;
	void *dc;
	size_t seltable_s;

	seltable_s = PICVIZ_SELECTION_NUMBER_OF_CHUNKS * sizeof(uint32_t);

	cudaMalloc((void**)&da, seltable_s);
	cudaMalloc((void**)&db, seltable_s);
	cudaMalloc((void**)&dc, seltable_s);

	cudaMemcpy(da, a->table, seltable_s, cudaMemcpyHostToDevice);
	cudaMemcpy(db, b->table, seltable_s, cudaMemcpyHostToDevice);

	picviz_selection_AB2C_or_k<<<6,256>>>((uint32_t *)da, (uint32_t *)db, (uint32_t *)dc);
	

	cudaMemcpy(a->table, dc, seltable_s, cudaMemcpyDeviceToHost);

	cudaFree(da);
	cudaFree(db);
	cudaFree(dc);
	
//	pv_row i;
//
//	for (i=0; i < PICVIZ_SELECTION_NUMBER_OF_CHUNKS; i++) {
//		a->table[i] = a->table[i] | b->table[i];
//	}
}


/******************************************************************************
 *
 * picviz_selection_AB2A_substraction
 *
 *****************************************************************************/
void picviz_selection_AB2A_substraction(picviz_selection_t *a, picviz_selection_t *b)
{
	pv_row i;

	for (i=0; i < PICVIZ_SELECTION_NUMBER_OF_CHUNKS; i++) {
		a->table[i] = a->table[i] & ~(b->table[i]);
	}
}


/******************************************************************************
 *
 * picviz_selection_AB2A_xor
 *
 *****************************************************************************/
void picviz_selection_AB2A_xor(picviz_selection_t *a, picviz_selection_t *b)
{
	pv_row i;

	for (i=0; i < PICVIZ_SELECTION_NUMBER_OF_CHUNKS; i++) {
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
 * picviz_selection_AB2C_and
 *
 *****************************************************************************/
void picviz_selection_AB2C_and(picviz_selection_t *a, picviz_selection_t *b, picviz_selection_t *c)
{
	pv_row i;

	for (i=0; i < PICVIZ_SELECTION_NUMBER_OF_CHUNKS; i++) {
		c->table[i] = a->table[i] & b->table[i];
	}
}


/******************************************************************************
 *
 * picviz_selection_AB2C_or
 *
 *****************************************************************************/
void picviz_selection_AB2C_or(picviz_selection_t *a, picviz_selection_t *b, picviz_selection_t *c)
{
	pv_row i;

	for (i=0; i < PICVIZ_SELECTION_NUMBER_OF_CHUNKS; i++) {
		c->table[i] = a->table[i] | b->table[i];
	}
}


/******************************************************************************
 *
 * picviz_selection_AB2C_substraction
 *
 *****************************************************************************/
void picviz_selection_AB2C_substraction(picviz_selection_t *a, picviz_selection_t *b, picviz_selection_t *c)
{
	pv_row i;

	for (i=0; i < PICVIZ_SELECTION_NUMBER_OF_CHUNKS; i++) {
		c->table[i] = a->table[i] & ~(b->table[i]);
	}
}


/******************************************************************************
 *
 * picviz_selection_AB2C_xor
 *
 *****************************************************************************/
void picviz_selection_AB2C_xor(picviz_selection_t *a, picviz_selection_t *b, picviz_selection_t *c)
{
	pv_row i;

	for (i=0; i < PICVIZ_SELECTION_NUMBER_OF_CHUNKS; i++) {
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
	picviz_selection_t *selection;

	selection = picviz_selection_new();
	printf("Line 12345 selection status: '%d'\n", picviz_selection_get_line(selection, 12345));
	picviz_selection_set_line(selection, 12345, 1);
	printf("Line 12345 selection status: '%d'\n", picviz_selection_get_line(selection, 12345));

	return 0;
}

int main(void)
{
	return test_selection();
}
#endif
