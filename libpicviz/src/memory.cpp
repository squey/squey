/**
 * \file memory.cpp
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#include <stdlib.h>

#include <picviz/general.h>
#include <picviz/memory.h>

#ifdef USE_VALGRIND
#include <valgrind/memcheck.h>
#endif

void *picviz_malloc(size_t size)
{
        return malloc(size);
}

void *picviz_realloc(void *ptr, size_t size)
{
        void *retptr;
	retptr = realloc(ptr, size);
        return retptr;
}

void picviz_free(void *ptr)
{
        return free(ptr);
}

