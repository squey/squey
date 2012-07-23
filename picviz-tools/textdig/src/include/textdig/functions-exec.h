/**
 * \file functions-exec.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef _FUNCTIONS_EXEC_H_
#define _FUNCTIONS_EXEC_H_

#include <stdio.h>
#include <stdarg.h>

#include <textdig/textdig.h>

#ifdef __cplusplus
 extern "C" {
#endif

void *function_run_from_string(const char *string);

#ifdef __cplusplus
 }
#endif


#endif /* _FUNCTIONS_EXEC_H_ */

