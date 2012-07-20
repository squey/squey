#ifndef PVCORE_PICVIZASSERT_H
#define PVCORE_PICVIZASSERT_H

#define ASSERT_VALID(b) _ASSERT_VALID(b, __FILE__, __LINE__)
#define _ASSERT_VALID(b, f, l)\
	do {\
	    if (!(b)) {\
			        PVLOG_ERROR("Check in %s:%d returned false: %s.\n", f, l, #b);\
			        assert(false);\
			        exit(1);\
		}\
	} while(0);

#endif
