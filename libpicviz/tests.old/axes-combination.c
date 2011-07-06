#include <picviz/general.h>
#include <picviz/init.h>
#include <picviz/axes-combination.h>

int main(void)
{
  picviz_axes_combination_t *axes_combination;

  picviz_init(0, NULL);

  axes_combination = picviz_axes_combination_new();
  picviz_axes_combination_destroy(axes_combination);

  picviz_terminate();

  return 0;
}
