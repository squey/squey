#include <stdio.h>

#include <picviz/general.h>
#include <picviz/debug.h>
#include <picviz/utils.h>

int i = 0;

int read_lines_function(long offset, char *line, size_t linesize, void *userdata)
{
  /* printf("Line %d:[%X[[%s]]]\n", i, line[1], line); */
  printf("line=[[[%s]]]\n", line);
  i++;
  return 0;
}

int main(void)
{
  FILE *fp;
  int ret = 1;
  long line_nb;

  fp = fopen("bluecoat.log", "rb");
  if (!fp) {
	printf("Cannot find the file\n");
	return 1;
  }

  line_nb = picviz_file_count_lines_from_fd(fp);
  printf("We have %d lines\n", line_nb);

  ret = picviz_file_line_foreach(fp, read_lines_function, 0, 0, NULL);

  fclose(fp);

  return 0;
}
