/*
 * $Id$
 * Copyright (C) Sebastien Tricaud 2010-2011
 * Copyright (C) Philippe Saade 2010-2011
 * 
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

