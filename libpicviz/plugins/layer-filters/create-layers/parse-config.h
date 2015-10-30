/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PICVIZ_CREATELAYERS_PARSECONFIG_H
#define PICVIZ_CREATELAYERS_PARSECONFIG_H

#include <QString>
#include <QStringList>
#include <QMap>

int create_layers_parse_config(QString filename, int (*handle_create_layers_section)(QString section_name, QMap<QString, QStringList> layers_regex));

#endif	/* PICVIZ_CREATELAYERS_PARSECONFIG_H */
