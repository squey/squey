#include <stdio.h>

#include <apr_general.h>
#include <apr_hash.h>

#include <picviz/general.h>
#include <picviz/debug.h>
#include <picviz/datatreerootitem.h>
#include <picviz/scene.h>
#include <picviz/source.h>
#include <picviz/format.h>
#include <picviz/normalize.h>

int main(void)
{
	picviz_datatreerootitem_t *datatree;
	picviz_scene_t *scene;
	picviz_source_t *source;
	picviz_format_t *format;
	picviz_normalize_functions_t *nfunctions;


	char *format_buffer;
	format_buffer = " \n\
 revision = 1 \n\
 \n\
 time-format[3] = \"%d/%b/%Y:%H:%M:%S %Z\"\n\
 \n \
 axes { \n\
      ipv4 default default \"Source IP\" \n\
      ipv4 default default \"Dest IP\" \n\
      time 24h default \"Time\" \n\
      integer default default \"Source Port\" \n\
      integer default default \"Dest Port\" \n\
      integer default default \"wft1\" \n\
      integer default default \"wft2\" \n\
      integer default default \"wft3\" \n\
      integer default default \"wft4\" \n\
 }\n \
 ";
#include "test-env.h"

	picviz_init(0, NULL);

	datatree = picviz_datatreerootitem_new();
	printf("Datatree created\n");
	scene = picviz_scene_new(datatree, "default");
	printf("Scene created\n");
	source = picviz_source_new(scene);
	printf("Source created\n");
	format = picviz_format_create_from_buffer(format_buffer); 
	picviz_format_debug(format);
	picviz_format_destroy(format);

	source->sourcetype = PICVIZ_STATIC_FILE;
	source->logtype = strdup("pcre:syslog");
	source->sourcename = strdup("test_petit.log");

	picviz_debug(PICVIZ_DEBUG_DEBUG, "Before getting the hash\n");

	printf("log plugin=%s\n", source->logplugin);

	/* picviz_debug(PICVIZ_DEBUG_DEBUG, "datatree->normalization_plugins(%X), source->logplugin(%s), strlen(source->logplugin)(%d)\n", datatree->normalization_plugins, source->logplugin, strlen(source->logplugin)); */

	nfunctions = apr_hash_get(datatree->normalization_plugins, source->logplugin, strlen(source->logplugin));
	if (!nfunctions) {
		picviz_debug(PICVIZ_DEBUG_CRITICAL, "Cannot get the plugin '%s' for filename '%s'\n", source->logtype, "test_petit.log");
		return NULL;
	}
	picviz_debug(PICVIZ_DEBUG_DEBUG, "After getting the hash\n");

#if 0
	picviz_debug(PICVIZ_DEBUG_DEBUG, "Before running picviz_normalize_get_format_function\n");
	source->format = nfunctions->formatfunc(source->logopt);
	picviz_debug(PICVIZ_DEBUG_DEBUG, "After running picviz_normalize_get_format_function\n");

	picviz_debug(PICVIZ_DEBUG_DEBUG, "Before running picviz_normalize_file_function\n");
	nraw = nfunctions->filefunc(source, filename);
	picviz_debug(PICVIZ_DEBUG_DEBUG, "After running picviz_normalize_file_function\n");
#endif

	picviz_terminate();

	return 0;
}

