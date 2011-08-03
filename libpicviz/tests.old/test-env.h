#ifdef WIN32
	putenv("PICVIZ_PARSERS_DIR=..\\..\\plugins\\parsers");
	putenv("PICVIZ_NORMALIZE_DIR=..\\..\\plugins\\normalize\\RelWithDebInfo");
	putenv("PICVIZ_FUNCTIONS_DIR=..\\..\\plugins\\functions\\RelWithDebInfo");
	putenv("PICVIZ_FILTERS_DIR=..\\..\\plugins\\filters\\RelWithDebInfo");
	putenv("PICVIZ_DEBUG_LEVEL=DEBUG");
#else
	setenv("PVRUSH_NORMALIZE_DIR","../../libpvkernel/rush/plugins/normalize",0);
	setenv("PVRUSH_NORMALIZE_HELPERS_DIR","../../libpvkernel/rush/plugins/normalize-helpers",0);

	setenv("PICVIZ_PARSERS_DIR","../plugins/parsers",0);
	setenv("PICVIZ_NORMALIZE_DIR","../plugins/normalize",0);
	setenv("PICVIZ_NORMALIZE_HELPERS_DIR","../plugins/normalize-helpers",0);
	setenv("PICVIZ_FUNCTIONS_DIR","../plugins/functions",0);
	setenv("PICVIZ_FILTERS_DIR","../plugins/filters",0);
	setenv("PICVIZ_DEBUG_LEVEL","DEBUG",0);
#endif

