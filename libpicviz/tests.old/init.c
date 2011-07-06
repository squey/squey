#include <stdio.h>

int main(int argc, char **argv)
{
	picviz_context_t *context;

	printf("Testing %s\n", __FUNCTION__);

	picviz_init(argc, NULL);

	picviz_terminate();

	return 0;
}
