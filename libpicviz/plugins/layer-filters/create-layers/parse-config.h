/**
 * \file parse-config.h
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#ifndef PICVIZ_CREATELAYERS_PARSECONFIG_H
#define PICVIZ_CREATELAYERS_PARSECONFIG_H

#include <QString>
#include <QStringList>
#include <QMap>

int create_layers_parse_config(QString filename, int (*handle_create_layers_section)(QString section_name, QMap<QString, QStringList> layers_regex));

#endif	/* PICVIZ_CREATELAYERS_PARSECONFIG_H */
