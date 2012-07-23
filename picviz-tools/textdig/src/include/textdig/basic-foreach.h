/**
 * \file basic-foreach.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef _BASIC_FOREACH_H_
#define _BASIC_FOREACH_H_


#ifdef __cplusplus
 extern "C" {
#endif

int print_line_x_function(long offset, char *line, size_t linesize, void *userdata);
int split_lines_function(long offset, char *line, size_t linesize, void *userdata);

#ifdef __cplusplus
 }
#endif


#endif /* _BASIC_FOREACH_H_ */

