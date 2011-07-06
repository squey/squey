#include <stdio.h>

#include <picviz/array.h>

int main(void)
{
	picviz_array_t *array1;
	float value;

	array1 = picviz_array_new(PICVIZ_ARRAY_STRING);
/* 	picviz_array_get(array1, 1, 1) = (float)12.3; */
/* 	value = picviz_array_get(array1, 1, 1) = 12.3; */
/* 	printf("value=%f\n", value); */
	picviz_array_destroy(array1);

	return 0;
}
