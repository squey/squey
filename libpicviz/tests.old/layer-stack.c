#include <picviz/layer-stack.h>

#include <stdio.h>
#include <stdlib.h>

#include <picviz/layer.h>
#include <picviz/view.h>


int main(int argc, char **argv)
{
	picviz_layer_stack_t *ls;
	picviz_layer_stack_t *ls2;
	picviz_layer_stack_t *ls3;
	picviz_layer_stack_t *ls4;

	picviz_layer_t *layer;
	picviz_layer_t *layer2;
	picviz_layer_t *layer3;
	

	int i, j_start;
	int line_index;
	int good;


	/**********************************************************************
	*
	* We test creation and deletion of a picviz_layer_stack_t
	*
	**********************************************************************/
	printf("\nWe test picviz_layer_stack_new and picviz_layer_stack_destroy\n");
	
	
	ls = picviz_layer_stack_new((picviz_view_t *)NULL);
	picviz_layer_stack_destroy(ls);

	/* good = 1; */
	/* if (!good) */
	/* { */
	/* 	printf("Layer Stack test : [%d] : picviz_layer_stack_new() or picviz_layer_stack_destroy() : failed\n", __LINE__); */
	/* 	return 1; */
	/* } */
	/* good = 0; */
	
	
	/**********************************************************************
	*
	* We test picviz_layer_stack_append_layer() and again picviz_layer_stack_destroy()
	*
	**********************************************************************/
	printf("\nWe test picviz_ layer_stack_append_layer() and again picviz_layer_stack_destroy()\n");
	ls = picviz_layer_stack_new((picviz_view_t *)NULL);
	layer = picviz_layer_new("Layer 1");
	
	picviz_layer_stack_append_layer(ls, layer);
	
	picviz_layer_stack_destroy(ls);
	
	/* good = 1; */
	/* if (!good) */
	/* { */
	/* 	printf("Layer Stack test : [%d] : picviz_layer_stack_append_layer() or picviz_layer_stack_destroy() : failed\n", __LINE__); */
	/* 	return 1; */
	/* } */
	/* good = 0; */
	
	
	/**********************************************************************
	*
	* We test picviz_layer_stack_append_new_layer() and again picviz_layer_stack_destroy()
	*
	**********************************************************************/
	printf("\nWe test picviz_ layer_stack_append_new_layer() and again picviz_layer_stack_destroy()\n");
	ls = picviz_layer_stack_new((picviz_view_t *)NULL);
	
	picviz_layer_stack_append_new_layer(ls);
	
	picviz_layer_stack_destroy(ls);
	
	/* good = 1; */
	/* if (!good) */
	/* { */
	/* 	printf("Layer Stack test : [%d] : picviz_layer_stack_append_new_layer() or picviz_layer_stack_destroy() : failed\n", __LINE__); */
	/* 	return 1; */
	/* } */
	/* good = 0; */

	
	/**********************************************************************
	*
	* We test picviz_layer_stack_move_layer_up() and picviz_layer_stack_move_layer_down()
	*
	**********************************************************************/
	
	printf("\nWe test picviz_layer_stack_append_new_layer() and again picviz_layer_stack_destroy()\n");
	ls = picviz_layer_stack_new((picviz_view_t *)NULL);
	layer = picviz_layer_new("A");
	layer2 = picviz_layer_new("B");
	layer3 = picviz_layer_new("C");
	
	picviz_layer_stack_append_layer(ls, layer);
	picviz_layer_stack_append_layer(ls, layer2);
	picviz_layer_stack_append_layer(ls, layer3);
	
	picviz_layer_stack_move_layer_up(ls,1);
	picviz_layer_stack_move_layer_down(ls,1);
	
	printf("This is the name of the bottom layer (should be C) : %s\n", ls->table[0]->name);
	printf("This is the name of the middle layer (should be A) : %s\n", ls->table[1]->name);
	printf("This is the name of the top layer (should be B) : %s\n", ls->table[2]->name);
	
	
	good = (!strcmp(ls->table[0]->name, "C")) && (!strcmp(ls->table[1]->name, "A")) && (!strcmp(ls->table[2]->name, "B"));
	if (!good)
	{
		printf("Layer Stack test : [%d] : picviz_layer_stack_layer_move_up() or picviz_layer_stack_layer_move_down() : failed\n", __LINE__);
		return 1;
	}
	
	
	
	
	printf("\n\n TEST IS OVER WITH SUCCESS !!! \n\n");

	return 0;
}
