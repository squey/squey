//! \file memory.cpp
//! $Id: memory.cpp 2520 2011-04-30 12:26:48Z stricaud $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

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

