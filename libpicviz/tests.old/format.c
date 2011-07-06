#include <stdio.h>
#include <unistd.h>

#include <picviz/general.h>
#include <picviz/init.h>
#include <picviz/format.h>

int main(int argc, char **argv)
{
	picviz_format_t *format;
	char *format_buffer;
	char *time_format;
	int retval;

#include "test-env.h"

    format_buffer = " \n\
 revision = 1\n\
 \n\
 pcre = \"(\\w+\\s+\\d+\\s+\\d+:\\d+:\\d+) (\\S+) ([a-zA-Z/]+)\\S*: (.*)\"\
 key-axes = \"1,2,5\"\
 time-format[3] = \"%d/%b/%Y:%H:%M:%S %Z\n\"\
 \n \
 axes { \n\
      \"Source IP\" ipv4 default default \n\
      \"Dest IP\" ipv4 default default \n\
      \"Time\" time 24h default \n\
      \"Source Port\" integer default default \n\
      \"Dest Port\" integer default default \n\
      \"wft1\" integer default default\n\
      \"wft2\" integer default default \n\
      \"wft3\" integer default default \n\
      \"wft4\" integer default default \n\
 }\n \
\n \
pcre-ignore {\n\
	\"kernel.*\" 3\n\
	\".*ssh.*\" 3\n\
}\n\
 ";


	picviz_init(argc, NULL);

 	format = picviz_format_create_from_buffer(format_buffer);

	picviz_format_debug(format);


	/* time_format = picviz_format_get_time_format(format, 3); */
	/* if (!time_format) return 1; */
	/* printf("Time format for axis 3: '%s'\n", time_format); */

	/* picviz_format_destroy(format); */

	/* format = picviz_format_new_from_buffer(format_buffer);  */
	/* picviz_format_destroy(format); */

	/* format = picviz_format_new_from_buffer(format_buffer);  */
	/* picviz_format_destroy(format); */

	/* format = picviz_format_new_from_buffer(format_buffer);  */
	picviz_format_destroy(format);

	picviz_terminate();

	return 0;
}
