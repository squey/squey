/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef _READ_FILE_H_
#define _READ_FILE_H_

#include <inendi/file.h>
#include <inendi/source.h>
#include <textdig/textdig.h>

#ifdef __cplusplus
 extern "C" {
#endif

inendi_source_t *textdig_read_file_get_source(inendi_file_t *file, textdig_options_t options);
void textdig_read_file_source_to_csv(inendi_source_t *source);

#ifdef __cplusplus
 }
#endif


#endif /* _READ_FILE_H_ */
