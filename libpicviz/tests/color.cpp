/**
 * \file color.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <stdio.h>
#include <math.h>

#include <apr_tables.h>

#include <picviz/general.h>
#include <picviz/color.h>



int main(int argc, char **argv)
{
	picviz_color_t *color1, *color2;
	apr_byte_t r_value;
	apr_byte_t g_value;
	apr_byte_t b_value;
	apr_byte_t a_value;
	float      h_value;
	float      s_value;
	float      v_value;
	int good;



	/**********************************************************************
	*
	* We test CONSTRUCTOR/DESTRUCTOR
	*
	**********************************************************************/
	printf("We test CONSTRUCTOR/DESTRUCTOR for picviz_color_t\n");
	color1 = picviz_color_new();
	picviz_color_destroy(color1);
	
	

	/**********************************************************************
	*
	* We test picviz_color_new and all picviz_color_get_[r,g,b,a,h,s,v]
	*
	**********************************************************************/
	printf("\nWe test picviz_color_get_[r,g,b,a,h,s,v]\n");
	color1 = picviz_color_new();
	r_value = picviz_color_get_r(color1);
	printf("This is the R value of a just-created color (should be 0): %d\n", r_value);
	g_value = picviz_color_get_g(color1);
	printf("This is the G value of a just-created color (should be 0): %d\n", g_value);
	b_value = picviz_color_get_b(color1);
	printf("This is the B value of a just-created color (should be 0): %d\n", b_value);
	a_value = picviz_color_get_a(color1);
	printf("This is the A value of a just-created color (should be 255): %d\n", a_value);
	h_value = picviz_color_get_h(color1);
	printf("This is the H value of a just-created color (should be 0): %f\n", h_value);
	s_value = picviz_color_get_s(color1);
	printf("This is the S value of a just-created color (should be 0): %f\n", s_value);
	v_value = picviz_color_get_v(color1);
	printf("This is the V value of a just-created color (should be 0): %f\n", v_value);
	
	good = (r_value == 0) && (g_value == 0) && (b_value == 0) && (a_value == 255) && (h_value == 0) && (s_value == 0) && (v_value == 0);
	if (!good)
	{
		printf("Color test : [%d] : picviz_color_new or picviz_color_get_[r,g,b,a,h,s,v] : failed\n", __LINE__);
		return 1;
	}



	/**********************************************************************
	*
	* We test all picviz_color_set_[r,g,b,a,h,s,v]
	*
	**********************************************************************/
	printf("\nWe test picviz_color_set_[r,g,b,a,h,s,v]\n");
	picviz_color_set_r(color1, 15);
	r_value = picviz_color_get_r(color1);
	printf("The R value of our color should be 15 : %d\n", r_value);
	
	picviz_color_set_g(color1, 16);
	g_value = picviz_color_get_g(color1);
	printf("The G value of our color should be 16 : %d\n", g_value);
	
	picviz_color_set_b(color1, 17);
	b_value = picviz_color_get_b(color1);
	printf("The B value of our color should be 17 : %d\n", b_value);
	
	picviz_color_set_a(color1, 18);
	a_value = picviz_color_get_a(color1);
	printf("The A value of our color should be 18 : %d\n", a_value);
	
	picviz_color_set_h(color1, 19.19);
	h_value = picviz_color_get_h(color1);
	printf("The H value of our color should be 19.19 : %f\n",h_value);
	
	picviz_color_set_s(color1, 0.20);
	s_value = picviz_color_get_s(color1);
	printf("The S value of our color should be 0.20 : %f\n", s_value);
	
	picviz_color_set_v(color1, 0.30);
	v_value = picviz_color_get_v(color1);
	printf("The S value of our color should be 0.30 : %f\n", v_value);
	

	good = (r_value == 15) && (g_value == 16) && (b_value == 17) && (a_value == 18) && (fabs(h_value - 19.19) < 0.001) && (fabs(s_value - 0.20) < 0.001) && (fabs(v_value - 0.30) < 0.001);
	if (!good)
	{
		printf("Color test : [%d] : picviz_color_set_[r,g,b,a,h,s,v] : failed\n", __LINE__);
		return 1;
	}

	
	
	/**********************************************************************
	*
	* We test picviz_color_set_rgba
	*
	**********************************************************************/
	printf("\nWe test picviz_color_set_rgba\n");
	picviz_color_set_rgba(color1, 11, 22, 33, 44);
	r_value = picviz_color_get_r(color1);
	printf("This is the R value of a just-created color (should be 11): %d\n", r_value);
	g_value = picviz_color_get_g(color1);
	printf("This is the G value of a just-created color (should be 22): %d\n", g_value);
	b_value = picviz_color_get_b(color1);
	printf("This is the B value of a just-created color (should be 33): %d\n", b_value);
	a_value = picviz_color_get_a(color1);
	printf("This is the A value of a just-created color (should be 44): %d\n", a_value);

	good = (r_value == 11) && (g_value == 22) && (b_value == 33) && (a_value == 44);
	if (!good) {
		printf("Color test : [%d] : picviz_color_set_rgba : failed\n", __LINE__);
		return 1;
	}



	/**********************************************************************
	*
	* We test picviz_color_set_hsv
	*
	**********************************************************************/
	printf("\nWe test picviz_color_set_hsv\n");
	picviz_color_set_hsv(color1, 120, 0.5, 0.5);
	h_value = picviz_color_get_h(color1);
	printf("This is the H value of the test (should be 120): %f\n", h_value);
	s_value = picviz_color_get_s(color1);
	printf("This is the S value of the test (should be 0.5): %f\n", s_value);
	v_value = picviz_color_get_v(color1);
	printf("This is the V value of the test (should be 0.5): %f\n", v_value);

	good = (fabs(h_value - 120.0) < 0.001) && (fabs(s_value - 0.5) < 0.001) && (fabs(v_value - 0.5) < 0.001);
	if (!good) {
		printf("Color test : [%d] : picviz_color_set_hsv : failed\n", __LINE__);
		return 1;
	}



	/**********************************************************************
	*
	* We test picviz_color_A2B_copy
	*
	**********************************************************************/
	printf("\nWe test picviz_color_A2B_copy\n");
	color2 = picviz_color_new();
	picviz_color_A2B_copy(color1, color2);
	
	good = (color1->r == color2->r) && (color1->g == color2->g) && (color1->b == color2->b) && (color1->a == color2->a) && (fabs(color1->h - color2->h) < 0.001) && (fabs(color1->s - color2->s) < 0.001) && (fabs(color1->v - color2->v) < 0.001);
	picviz_color_destroy(color2);
	
	if (!good) {
		printf("Color test : [%d] : picviz_color_A2B_copy : failed\n", __LINE__);
		return 1;
	}



	/**********************************************************************
	*
	* We test picviz_color_hsv_to_rgb
	*
	**********************************************************************/
	printf("\nWe test picviz_color_hsv_to_rgb \n");
	picviz_color_destroy(color1);
	color1 = picviz_color_new();
	
	picviz_color_set_hsv(color1, 201, 0.50196, 0.862745);
	picviz_color_hsv_to_rgb(color1);
	r_value = picviz_color_get_r(color1);
	printf("This is the R value (should be 109): %d\n", r_value);
	g_value = picviz_color_get_g(color1);
	printf("This is the G value (should be 181): %d\n", g_value);
	b_value = picviz_color_get_b(color1);
	printf("This is the B value (should be 219): %d\n", b_value);
	a_value = picviz_color_get_a(color1);
	printf("This is the A value (should be 255): %d\n", a_value);
	h_value = picviz_color_get_h(color1);
	printf("This is the H value (should be 201.00000): %f\n", h_value);
	s_value = picviz_color_get_s(color1);
	printf("This is the S value (should be 0.501960): %f\n", s_value);
	v_value = picviz_color_get_v(color1);
	printf("This is the V value (should be 0.862745): %f\n", v_value);

	good = (r_value == 109) && (g_value == 181) && (b_value == 219) && (a_value == 255);
	if (!good) {
		printf("Color test : [%d] : picviz_color_hsv_to_rgb : failed\n", __LINE__);
		return 1;
	}



	/**********************************************************************
	*
	* We test picviz_color_rgb_to_hsv
	*
	**********************************************************************/
	printf("\nWe try to set RGB(51, 40, 102) as an RBGA color, apply picviz_color_rgb_to_hsv and test all the values\n");
	picviz_color_set_rgba(color1, 51, 40, 102, 255);
	picviz_color_rgb_to_hsv(color1);
	r_value = picviz_color_get_r(color1);
	printf("This is the R value (should be 51): %d\n", r_value);
	g_value = picviz_color_get_g(color1);
	printf("This is the G value (should be 40): %d\n", g_value);
	b_value = picviz_color_get_b(color1);
	printf("This is the B value (should be 102): %d\n", b_value);
	a_value = picviz_color_get_a(color1);
	printf("This is the A value (should be 255): %d\n", a_value);
	h_value = picviz_color_get_h(color1);
	printf("This is the H value (should be 250.645157): %f\n", h_value);
	s_value = picviz_color_get_s(color1);
	printf("This is the S value (should be 0.607843): %f\n", s_value);
	v_value = picviz_color_get_v(color1);
	printf("This is the V value (should be 0.400000): %f\n", v_value);

	good = (fabs(h_value - 250.645157) < 0.001) && (fabs(s_value - 0.607843) < 0.001) && (fabs(v_value - 0.400000) < 0.001);
	if (!good) {
		printf("Color test : [%d] : picviz_color_rgb_to_hsv : failed\n", __LINE__);
		return 1;
	}



	/**********************************************************************
	*
	* We test picviz_color_extract_[r/g/b]_from_html_code
	*
	**********************************************************************/
	printf("\nWe test the picviz_color_extract_[r/g/b]_from_html_code functions\n");
	printf("We use the #326496 Hex color\n");
	r_value = picviz_color_extract_r_from_html_code("#326496");
	printf("This is the R value (should be 50): %d\n", r_value);
	g_value = picviz_color_extract_g_from_html_code("#326496");
	printf("This is the G value (should be 100): %d\n", g_value);
	b_value = picviz_color_extract_b_from_html_code("#326496");
	printf("This is the B value (should be 150): %d\n", b_value);
	
	good = (r_value == 50) && (g_value == 100) && (b_value == 150);
	if (!good) {
		printf("Color test : [%d] : picviz_color_extract_[r/g/b]_from_html_code : failed\n", __LINE__);
		return 1;
	}





	printf("\n\n TEST IS OVER WITH SUCCESS !!! \n\n");

	return 0;
}