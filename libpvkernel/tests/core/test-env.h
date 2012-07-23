/**
 * \file test-env.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifdef WIN32
#else
	setenv("PVCORE_SHARE_DIR","../share",0);
//	setenv("PICVIZ_DEBUG_FILE","out.txt",0);
#endif

