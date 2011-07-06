#include <stdio.h>

#include <apr_tables.h>

#include <picviz/axis.h>
#include <picviz/axes-combination.h>
#include <picviz/view.h>
#include <picviz/layer.h>
#include <picviz/selection.h>
#include <picviz/line-properties.h>
#include <picviz/filter-library.h>
#include <picviz/utils.h>

#define LOGTYPE "pcap:netflow"
#define LOGFILE "single.pcap"

int main(int argc, char **argv)
{
	picviz_view_t *view;

	setenv("PICVIZ_NORMALIZE_DIR","../plugins/normalize/",0);

	picviz_init(argc, argv);

	/* view = picviz_h_view_create(context, LOGTYPE, LOGFILE); */

	picviz_terminate();

	return 0;
}
