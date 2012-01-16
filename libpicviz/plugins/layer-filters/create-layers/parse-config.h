//! \file parse-config.h
//! $Id$
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PICVIZ_CREATELAYERS_PARSECONFIG_H
#define PICVIZ_CREATELAYERS_PARSECONFIG_H

#include <QString>

int create_layers_parse_config(QString filename, int (*handle_create_layers_section)(QString section_name, QString layer_name, QString regex_for_layer));

#endif	/* PICVIZ_CREATELAYERS_PARSECONFIG_H */
