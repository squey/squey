#include <stdlib.h>


#ifdef WIN32
#define pv_setenv(a,b,c) putenv(a "=" b)
#else
#define pv_setenv(a,b,c) setenv(a,b,c)
#endif

void init_env()
{
	pv_setenv("PVFILTER_NORMALIZE_DIR","../../plugins/normalize",0);
	pv_setenv("PVRUSH_NORMALIZE_HELPERS_DIR","../../plugins/normalize-helpers:./test-formats",0);
	pv_setenv("PICVIZ_DEBUG_LEVEL","DEBUG",0);
	pv_setenv("PICVIZ_CACHE_DIR","./cache",0);
	pv_setenv("PVRUSH_INPUTTYPE_DIR","../../plugins/input_types",0);
	pv_setenv("PVRUSH_SOURCE_DIR","../../plugins/sources",0);
}
