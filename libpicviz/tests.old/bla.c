#include <stdio.h>
#include <stdlib.h>

#include <picviz/general.h>
#include <picviz/big-log-array.h>

int main(void)
{
  picviz_bla_t *bla;

#include "test-env.h"

  bla = picviz_bla_new("test_petit.log");

  picviz_bla_create_plaintext_index(bla, 100);

  /* printf("line 1 offset index:'%d'\n", bla->index[1]); */

  picviz_bla_destroy(bla);

  return 0;
}

