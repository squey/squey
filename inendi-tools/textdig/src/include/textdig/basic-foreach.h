/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef _BASIC_FOREACH_H_
#define _BASIC_FOREACH_H_

#ifdef __cplusplus
extern "C" {
#endif

int print_line_x_function(long offset, char* line, size_t linesize, void* userdata);
int split_lines_function(long offset, char* line, size_t linesize, void* userdata);

#ifdef __cplusplus
}
#endif

#endif /* _BASIC_FOREACH_H_ */
