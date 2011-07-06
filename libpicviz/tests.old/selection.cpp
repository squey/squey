#include <picviz/selection.h>

#include <stdio.h>

int main(int argc, char **argv)
{
	picviz_selection_t *selection;
	picviz_selection_t *selection2;
	picviz_selection_t *selection3;

	int i, good;
	int line_index;

	pv_row last_index;
	pv_row count;

	/**********************************************************************
	*
	* We test creation and deletion of a picviz_selection_t
	*
	**********************************************************************/

	selection = picviz_selection_new();
	picviz_selection_destroy(selection);


	/**********************************************************************
	*
	* We test all Generic functions
	*
	**********************************************************************/

	// we test picviz_selection_count and picviz_selection_A2A_select_all
	printf("we test picviz_selection_count and picviz_selection_A2A_select_all\n");
	last_index = 16777216 - 1;

	selection = picviz_selection_new();
	picviz_selection_A2A_select_all(selection);
	count = picviz_selection_count(selection, last_index);
	printf("is 16777216 = to %u ?\n\n", count);

	// we test picviz_selection_get_line and picviz_selection_A2A_select_even
	printf("we test picviz_selection_get_line and picviz_selection_A2A_select_even\n");
	picviz_selection_A2A_select_even(selection);
	count = picviz_selection_count(selection, last_index);
	printf("is 8388608 = to %u ?\n\n", count);
	
	// we test picviz_selection_get_line and picviz_selection_A2A_select_odd
	printf("we test picviz_selection_get_line and picviz_selection_A2A_select_odd\n");
	picviz_selection_A2A_select_odd(selection);
	count = picviz_selection_count(selection, last_index);
	printf("is 8388608 = to %u ?\n\n", count);


	// we test picviz_selection_get_line_id_at_index
	printf("we test picviz_selection_get_line_id_at_index\n");
	count = picviz_selection_get_line_index_of_nth_selected_line(selection, 8388608);
	printf("is 16777215 = to %u ?\n\n", count);


	// we test picviz_selection_get_number_of_selected_lines_in_range
	printf("we test picviz_selection_get_number_of_selected_lines_in_range\n");
	count = picviz_selection_get_number_of_selected_lines_in_range(selection, 0, 100);
	printf("is 50 = to %u ?\n\n", count);
	count = picviz_selection_get_number_of_selected_lines_in_range(selection, 0, 101);
	printf("is 50 = to %u ?\n\n", count);

	// FIXME! we do NOT test picviz_selection_mapped_foreach

	// FIXME! we do NOT test picviz_selection_nraw_foreach

	// FIXME! we do NOT test picviz_selection_plotted_foreach

	// FIXME! we do NOT test picviz_selection_set_line

	picviz_selection_destroy(selection);



	
	/**********************************************************************
	***********************************************************************
	*
	* We test all functions picviz_selection_A2A_****
	*
	***********************************************************************
	**********************************************************************/

	
	/**********************************************************************
	*
	* We test picviz_selection_A2A_inverse()
	*
	**********************************************************************/

	printf("\nWe test picviz_selection_A2A_inverse()\n");
	selection = picviz_selection_new();
	picviz_selection_A2A_inverse(selection);

	for (i=0; i<16777216; i++) {
		good = (picviz_selection_get_line(selection, i) == 1);
		if (!good) {
			printf(" i = %d, and state of line n is : %d\n", i, picviz_selection_get_line(selection, i));
			printf("selection test : [%d] : picviz_selection_A2A_inverse : failed\n", __LINE__);
			return 1;
		}
	}
	
	picviz_selection_destroy(selection);



	/**********************************************************************
	*
	* We test picviz_selection_A2A_select_all()
	*
	**********************************************************************/

	printf("\nWe test picviz_selection_A2A_select_all()\n");
	selection = picviz_selection_new();
	picviz_selection_A2A_select_all(selection);

	for (i=0; i<16777216; i++) {
		good = (picviz_selection_get_line(selection, i) == 1);
		if (!good) {
			printf(" i = %d, and state of line n is : %d\n", i, picviz_selection_get_line(selection, i));
			printf("selection test : [%d] : picviz_selection_A2A_select_all : failed\n", __LINE__);
			return 1;
		}
	}

	picviz_selection_destroy(selection);




	/**********************************************************************
	*
	* We test picviz_selection_A2A_select_even()
	*
	**********************************************************************/

	printf("\nWe test picviz_selection_A2A_select_even()\n");
	selection = picviz_selection_new();
	picviz_selection_A2A_select_even(selection);

	for (i=0; i<16777216; i += 2) {
		good = (picviz_selection_get_line(selection, i) == 1);
		if (!good) {
			printf(" (A) i = %d, and state of line n is : %d\n", i, picviz_selection_get_line(selection, i));
			printf("selection test : [%d] : picviz_selection_A2A_select_even : failed\n", __LINE__);
			return 1;
		}
	}


	for (i=1; i<16777216; i += 2) {
		good = (picviz_selection_get_line(selection, i) == 0);
		if (!good) {
			printf(" (B) i = %d, and state of line n is : %d\n", i, picviz_selection_get_line(selection, i));
			printf("selection test : [%d] : picviz_selection_A2A_select_even : failed\n", __LINE__);
			return 1;
		}
	}

	picviz_selection_destroy(selection);




	/**********************************************************************
	*
	* We test picviz_selection_A2A_select_none()
	*
	**********************************************************************/

	printf("\nWe test picviz_selection_A2A_select_none()\n");
	selection = picviz_selection_new();
	picviz_selection_A2A_select_none(selection);

	for (i=0; i<16777216; i++) {
		good = (picviz_selection_get_line(selection, i) == 0);
		if (!good) {
			printf(" i = %d, and state of line n is : %d\n", i, picviz_selection_get_line(selection, i));
			printf("selection test : [%d] : picviz_selection_A2A_select_none : failed\n", __LINE__);
			return 1;
		}
	}

	picviz_selection_destroy(selection);



	/**********************************************************************
	*
	* We test picviz_selection_A2A_select_odd()
	*
	**********************************************************************/

	printf("\nWe test picviz_selection_A2A_select_odd()\n");
	selection = picviz_selection_new();
	picviz_selection_A2A_select_odd(selection);

	for (i=0; i<16777216; i += 2) {
		good = (picviz_selection_get_line(selection, i) == 0);
		if (!good) {
			printf(" (A) i = %d, and state of line n is : %d\n", i, picviz_selection_get_line(selection, i));
			printf("selection test : [%d] : picviz_selection_A2A_select_odd : failed\n", __LINE__);
			return 1;
		}
	}


	for (i=1; i<16777216; i += 2) {
		good = (picviz_selection_get_line(selection, i) == 1);
		if (!good) {
			printf(" (B) i = %d, and state of line n is : %d\n", i, picviz_selection_get_line(selection, i));
			printf("selection test : [%d] : picviz_selection_A2A_select_odd : failed\n", __LINE__);
			return 1;
		}
	}

	picviz_selection_destroy(selection);







	


	selection = picviz_selection_new();
	picviz_selection_A2A_select_odd(selection);
	selection2 = picviz_selection_new();
	picviz_selection_A2A_select_even(selection2);
	picviz_selection_A2A_inverse(selection);
	count = 0;
	for (line_index = 0; line_index < PICVIZ_LINES_MAX; line_index++) {
		if (picviz_selection_get_line(selection, line_index) != picviz_selection_get_line(selection2, line_index)) {
			count++;
		}
	}
	if (count) {
		printf("FAILED : test of picviz_selection_A2A_inverse !\n\n");
	}
	else {
		printf("SUCCESS : test of picviz_selection_A2A_inverse !\n\n");
	}
	

	picviz_selection_destroy(selection);
	picviz_selection_destroy(selection2);




	/**********************************************************************
	***********************************************************************
	*
	* We test all functions picviz_selection_A2B_****
	*
	***********************************************************************
	**********************************************************************/

	/**********************************************************************
	*
	* We test picviz_selection_A2B_copy()
	*
	**********************************************************************/

	printf("\nWe test picviz_selection_A2B_copy()\n");
	selection = picviz_selection_new();
	selection2 = picviz_selection_new();
	picviz_selection_A2A_select_even(selection);
	picviz_selection_A2A_select_odd(selection2);
	picviz_selection_A2B_copy(selection, selection2);

	for (i=0; i<16777216; i += 2) {
		good = (picviz_selection_get_line(selection2, i) == 1);
		if (!good) {
			printf(" (A) i = %d, and state of line n is : %d\n", i, picviz_selection_get_line(selection2, i));
			printf("selection test : [%d] : picviz_selection_A2B_copy : failed\n", __LINE__);
			return 1;
		}
	}


	for (i=1; i<16777216; i += 2) {
		good = (picviz_selection_get_line(selection2, i) == 0);
		if (!good) {
			printf(" (B) i = %d, and state of line n is : %d\n", i, picviz_selection_get_line(selection2, i));
			printf("selection test : [%d] : picviz_selection_A2B_copy : failed\n", __LINE__);
			return 1;
		}
	}

	picviz_selection_destroy(selection);
	picviz_selection_destroy(selection2);



	/**********************************************************************
	*
	* We test picviz_selection_A2B_inverse()
	*
	**********************************************************************/

	printf("\nWe test picviz_selection_A2B_inverse()\n");
	selection = picviz_selection_new();
	selection2 = picviz_selection_new();
	picviz_selection_A2A_select_odd(selection);
	picviz_selection_A2A_select_odd(selection2);
	picviz_selection_A2B_inverse(selection, selection2);

	for (i=0; i<16777216; i += 2) {
		good = (picviz_selection_get_line(selection2, i) == 1);
		if (!good) {
			printf(" (A) i = %d, and state of line n is : %d\n", i, picviz_selection_get_line(selection2, i));
			printf("selection test : [%d] : picviz_selection_A2B_inverse() : failed\n", __LINE__);
			return 1;
		}
	}


	for (i=1; i<16777216; i += 2) {
		good = (picviz_selection_get_line(selection2, i) == 0);
		if (!good) {
			printf(" (B) i = %d, and state of line n is : %d\n", i, picviz_selection_get_line(selection2, i));
			printf("selection test : [%d] : picviz_selection_A2B_inverse() : failed\n", __LINE__);
			return 1;
		}
	}

	picviz_selection_destroy(selection);
	picviz_selection_destroy(selection2);








	/**********************************************************************
	***********************************************************************
	*
	* We test all functions picviz_selection_AB2A_****
	*
	***********************************************************************
	**********************************************************************/

	/**********************************************************************
	*
	* We test picviz_selection_AB2A_and()
	*
	**********************************************************************/

	printf("\nWe test picviz_selection_AB2A_and()\n");
	selection = picviz_selection_new();
	selection2 = picviz_selection_new();
	picviz_selection_A2A_select_all(selection);
	picviz_selection_A2A_select_even(selection2);
	picviz_selection_AB2A_and(selection, selection2);

	for (i=0; i<16777216; i += 2) {
		good = (picviz_selection_get_line(selection, i) == 1);
		if (!good) {
			printf(" (A) i = %d, and state of line n is : %d\n", i, picviz_selection_get_line(selection, i));
			printf("selection test : [%d] : picviz_selection_AB2A_and : failed\n", __LINE__);
			return 1;
		}
	}


	for (i=1; i<16777216; i += 2) {
		good = (picviz_selection_get_line(selection, i) == 0);
		if (!good) {
			printf(" (B) i = %d, and state of line n is : %d\n", i, picviz_selection_get_line(selection, i));
			printf("selection test : [%d] : picviz_selection_AB2A_and : failed\n", __LINE__);
			return 1;
		}
	}

	picviz_selection_destroy(selection);
	picviz_selection_destroy(selection2);



	/**********************************************************************
	*
	* We test picviz_selection_AB2A_or()
	*
	**********************************************************************/

	printf("\nWe test picviz_selection_AB2A_or()\n");
	selection = picviz_selection_new();
	selection2 = picviz_selection_new();
	picviz_selection_A2A_select_none(selection);
	picviz_selection_A2A_select_even(selection2);
	picviz_selection_AB2A_or(selection, selection2);

	for (i=0; i<16777216; i += 2) {
		good = (picviz_selection_get_line(selection, i) == 1);
		if (!good) {
			printf(" (A) i = %d, and state of line n is : %d\n", i, picviz_selection_get_line(selection, i));
			printf("selection test : [%d] : picviz_selection_AB2A_or : failed\n", __LINE__);
			return 1;
		}
	}


	for (i=1; i<16777216; i += 2) {
		good = (picviz_selection_get_line(selection, i) == 0);
		if (!good) {
			printf(" (B) i = %d, and state of line n is : %d\n", i, picviz_selection_get_line(selection, i));
			printf("selection test : [%d] : picviz_selection_AB2A_or : failed\n", __LINE__);
			return 1;
		}
	}

	picviz_selection_destroy(selection);
	picviz_selection_destroy(selection2);



	/**********************************************************************
	*
	* We test picviz_selection_AB2A_substraction()
	*
	**********************************************************************/

	printf("\nWe test picviz_selection_AB2A_substraction()\n");
	selection = picviz_selection_new();
	selection2 = picviz_selection_new();
	picviz_selection_A2A_select_all(selection);
	picviz_selection_A2A_select_odd(selection2);
	picviz_selection_AB2A_substraction(selection, selection2);

	for (i=0; i<16777216; i += 2) {
		good = (picviz_selection_get_line(selection, i) == 1);
		if (!good) {
			printf(" (A) i = %d, and state of line n is : %d\n", i, picviz_selection_get_line(selection, i));
			printf("selection test : [%d] : picviz_selection_AB2A_substraction : failed\n", __LINE__);
			return 1;
		}
	}


	for (i=1; i<16777216; i += 2) {
		good = (picviz_selection_get_line(selection, i) == 0);
		if (!good) {
			printf(" (B) i = %d, and state of line n is : %d\n", i, picviz_selection_get_line(selection, i));
			printf("selection test : [%d] : picviz_selection_AB2A_substraction : failed\n", __LINE__);
			return 1;
		}
	}

	picviz_selection_destroy(selection);
	picviz_selection_destroy(selection2);

	


	/**********************************************************************
	*
	* We test picviz_selection_AB2A_xor()
	*
	**********************************************************************/

	printf("\nWe test picviz_selection_AB2A_xor()\n");
	selection = picviz_selection_new();
	selection2 = picviz_selection_new();
	picviz_selection_A2A_select_all(selection);
	picviz_selection_A2A_select_odd(selection2);
	picviz_selection_AB2A_xor(selection, selection2);

	for (i=0; i<16777216; i += 2) {
		good = (picviz_selection_get_line(selection, i) == 1);
		if (!good) {
			printf(" (A) i = %d, and state of line n is : %d\n", i, picviz_selection_get_line(selection, i));
			printf("selection test : [%d] : picviz_selection_AB2A_xor : failed\n", __LINE__);
			return 1;
		}
	}


	for (i=1; i<16777216; i += 2) {
		good = (picviz_selection_get_line(selection, i) == 0);
		if (!good) {
			printf(" (B) i = %d, and state of line n is : %d\n", i, picviz_selection_get_line(selection, i));
			printf("selection test : [%d] : picviz_selection_AB2A_xor : failed\n", __LINE__);
			return 1;
		}
	}

	picviz_selection_destroy(selection);
	picviz_selection_destroy(selection2);












	
	/**********************************************************************
	***********************************************************************
	*
	* We test all functions picviz_selection_AB2C_****
	*
	***********************************************************************
	**********************************************************************/



	/**********************************************************************
	*
	* We test picviz_selection_AB2C_and()
	*
	**********************************************************************/

	printf("\nWe test picviz_selection_AB2C_and()\n");
	selection = picviz_selection_new();
	selection2 = picviz_selection_new();
	selection3 = picviz_selection_new();
	picviz_selection_A2A_select_all(selection);
	picviz_selection_A2A_select_even(selection2);
	picviz_selection_AB2C_and(selection, selection2, selection3);

	for (i=0; i<16777216; i += 2) {
		good = (picviz_selection_get_line(selection3, i) == 1);
		if (!good) {
			printf(" (A) i = %d, and state of line n is : %d\n", i, picviz_selection_get_line(selection3, i));
			printf("selection test : [%d] : picviz_selection_AB2C_and : failed\n", __LINE__);
			return 1;
		}
	}


	for (i=1; i<16777216; i += 2) {
		good = (picviz_selection_get_line(selection3, i) == 0);
		if (!good) {
			printf(" (B) i = %d, and state of line n is : %d\n", i, picviz_selection_get_line(selection3, i));
			printf("selection test : [%d] : picviz_selection_AB2C_and : failed\n", __LINE__);
			return 1;
		}
	}

	picviz_selection_destroy(selection);
	picviz_selection_destroy(selection2);
	picviz_selection_destroy(selection3);



	/**********************************************************************
	*
	* We test picviz_selection_AB2C_or()
	*
	**********************************************************************/

	printf("\nWe test picviz_selection_AB2C_or()\n");
	selection = picviz_selection_new();
	selection2 = picviz_selection_new();
	selection3 = picviz_selection_new();
	picviz_selection_A2A_select_none(selection);
	picviz_selection_A2A_select_even(selection2);
	picviz_selection_AB2C_or(selection, selection2, selection3);

	for (i=0; i<16777216; i += 2) {
		good = (picviz_selection_get_line(selection3, i) == 1);
		if (!good) {
			printf(" (A) i = %d, and state of line n is : %d\n", i, picviz_selection_get_line(selection3, i));
			printf("selection test : [%d] : picviz_selection_AB2C_or : failed\n", __LINE__);
			return 1;
		}
	}


	for (i=1; i<16777216; i += 2) {
		good = (picviz_selection_get_line(selection3, i) == 0);
		if (!good) {
			printf(" (B) i = %d, and state of line n is : %d\n", i, picviz_selection_get_line(selection3, i));
			printf("selection test : [%d] : picviz_selection_AB2C_or : failed\n", __LINE__);
			return 1;
		}
	}

	picviz_selection_destroy(selection);
	picviz_selection_destroy(selection2);
	picviz_selection_destroy(selection3);



	/**********************************************************************
	*
	* We test picviz_selection_AB2C_substraction()
	*
	**********************************************************************/

	printf("\nWe test picviz_selection_AB2C_substraction()\n");
	selection = picviz_selection_new();
	selection2 = picviz_selection_new();
	selection3 = picviz_selection_new();
	picviz_selection_A2A_select_all(selection);
	picviz_selection_A2A_select_odd(selection2);
	picviz_selection_AB2C_substraction(selection, selection2, selection3);

	for (i=0; i<16777216; i += 2) {
		good = (picviz_selection_get_line(selection3, i) == 1);
		if (!good) {
			printf(" (A) i = %d, and state of line n is : %d\n", i, picviz_selection_get_line(selection3, i));
			printf("selection test : [%d] : picviz_selection_AB2C_substraction : failed\n", __LINE__);
			return 1;
		}
	}


	for (i=1; i<16777216; i += 2) {
		good = (picviz_selection_get_line(selection3, i) == 0);
		if (!good) {
			printf(" (B) i = %d, and state of line n is : %d\n", i, picviz_selection_get_line(selection3, i));
			printf("selection test : [%d] : picviz_selection_AB2C_substraction : failed\n", __LINE__);
			return 1;
		}
	}

	picviz_selection_destroy(selection);
	picviz_selection_destroy(selection2);
	picviz_selection_destroy(selection3);




	/**********************************************************************
	*
	* We test picviz_selection_AB2C_xor()
	*
	**********************************************************************/

	printf("\nWe test picviz_selection_AB2C_xor()\n");
	selection = picviz_selection_new();
	selection2 = picviz_selection_new();
	selection3 = picviz_selection_new();
	picviz_selection_A2A_select_all(selection);
	picviz_selection_A2A_select_odd(selection2);
	picviz_selection_AB2C_xor(selection, selection2, selection3);

	for (i=0; i<16777216; i += 2) {
		good = (picviz_selection_get_line(selection3, i) == 1);
		if (!good) {
			printf(" (A) i = %d, and state of line n is : %d\n", i, picviz_selection_get_line(selection3, i));
			printf("selection test : [%d] : picviz_selection_AB2C_xor : failed\n", __LINE__);
			return 1;
		}
	}


	for (i=1; i<16777216; i += 2) {
		good = (picviz_selection_get_line(selection3, i) == 0);
		if (!good) {
			printf(" (B) i = %d, and state of line n is : %d\n", i, picviz_selection_get_line(selection3, i));
			printf("selection test : [%d] : picviz_selection_AB2C_xor : failed\n", __LINE__);
			return 1;
		}
	}

	picviz_selection_destroy(selection);
	picviz_selection_destroy(selection2);
	picviz_selection_destroy(selection3);








	return 0;
}
