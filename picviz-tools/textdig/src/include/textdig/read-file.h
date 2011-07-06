/*
 * $Id$
 * Copyright (C) Sebastien Tricaud 2010-2011
 * Copyright (C) Philippe Saade 2010-2011
 * 
 */

#ifndef _READ_FILE_H_
#define _READ_FILE_H_

#include <picviz/file.h>
#include <picviz/source.h>
#include <textdig/textdig.h>

#ifdef __cplusplus
 extern "C" {
#endif

picviz_source_t *textdig_read_file_get_source(picviz_file_t *file, textdig_options_t options);
void textdig_read_file_source_to_csv(picviz_source_t *source);

#ifdef __cplusplus
 }
#endif


#endif /* _READ_FILE_H_ */
