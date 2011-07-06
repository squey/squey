#include <picviz/line-properties.h>

#include <stdio.h>
#include <picviz/color.h>
#include <picviz/selection.h>



int main(int argc, char **argv)
{
	picviz_lines_properties_t *lp;
	picviz_lines_properties_t *lp2;
	picviz_lines_properties_t *lp3;
	picviz_lines_properties_t *lp4;

	picviz_color_t *color;

	picviz_selection_t *selection;
	

	int i, j_start;
	int line_index;
	int good;


	/**********************************************************************
	*
	* We test creation and deletion of a picviz_lines_properties_t
	*
	**********************************************************************/

	printf("\nWe test picviz_ lines_properties_new and picviz_lines_properties_destroy\n");
	lp = picviz_lines_properties_new();
	picviz_lines_properties_destroy(lp);

	good = 1;
	if (!good)
	{
		printf("Lines Properties test : [%d] : picviz_lines_properties_new() or _destroy() : failed\n", __LINE__);
		return 1;
	}





	/**********************************************************************
	*
	* We test picviz_lines_properties_line_set_rgba_from_color()
	*
	**********************************************************************/

	printf("\nWe test picviz_lines_properties_line_set_rgba_from_color\n");
	lp = picviz_lines_properties_new();
	color = picviz_color_new();
	
	for (i=0; i<256; i++) {
		picviz_color_set_rgba(color, i, i, i, i);
		picviz_lines_properties_line_set_rgba_from_color(lp, j_start + i, color);
	}

	for (i=0; i<256; i++) {
		good = (picviz_lines_properties_line_get_r(lp, j_start + i) == i); 
		if (!good) {
			printf("Lines Properties test : [%d] : picviz_lines_properties_line_set_rgba_from_color : failed\n", __LINE__);
			return 1;
		}
	}
	
	picviz_color_destroy(color);
	picviz_lines_properties_destroy(lp);



	

	/**********************************************************************
	*
	* We test picviz_lines_properties_selection_set_rgba_from_color()
	*
	**********************************************************************/

	printf("\nWe test picviz_lines_properties_selection_set_rgba_from_color\n");
	lp = picviz_lines_properties_new();
	color = picviz_color_new();
	picviz_color_set_rgba(color, 10, 20, 30, 40);

	selection = picviz_selection_new();
	picviz_selection_A2A_select_odd(selection);

	picviz_lines_properties_selection_set_rgba_from_color(lp, selection, 30000, color);

	for (i=1; i<30000; i += 2) {
		good = (picviz_lines_properties_line_get_r(lp, i) == 10) & (picviz_lines_properties_line_get_g(lp, i) == 20) & (picviz_lines_properties_line_get_b(lp, i) == 30) & (picviz_lines_properties_line_get_a(lp, i) == 40);
		if (!good) {
			printf(" (A) : Lines Properties test : [%d] : picviz_lines_properties_selection_set_rgba_from_color : failed\n", __LINE__);
			return 1;
		}
	}

	for (i=0; i<30000; i += 2) {
		good = (picviz_lines_properties_line_get_r(lp, i) == 255) & (picviz_lines_properties_line_get_g(lp, i) == 255) & (picviz_lines_properties_line_get_b(lp, i) == 255) & (picviz_lines_properties_line_get_b(lp, i) == 255);
		if (!good) {
			printf(" (B) : Lines Properties test : [%d] : picviz_lines_properties_selection_set_rgba_from_color : failed\n", __LINE__);
			return 1;
		}
	}
	
	picviz_color_destroy(color);
	picviz_lines_properties_destroy(lp);



	printf("\n\n TEST IS OVER WITH SUCCESS !!! \n\n");

	return 0;
}