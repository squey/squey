/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVFORMAT_VERSION_H
#define PVFORMAT_VERSION_H

#include <QDomDocument>

namespace PVRush
{

namespace PVFormatVersion
{
void to_current(QDomDocument& doc);

// Migration of mapping/plotting/type
QString const get_type_from_format(QString const& type_attr, QString const& mapped_attr);
QString const get_mapped_from_format(QString const& type_attr, QString const& mapped_attr);
QString const get_plotted_from_format(QString const& type_attr,
                                      QString const& mapped_attr,
                                      QString const& plotted_attr);

namespace __impl
{
void from0to1(QDomDocument& doc);
void from1to2(QDomDocument& doc);
void from2to3(QDomDocument& doc);
void from3to4(QDomDocument& doc);
void from4to5(QDomDocument& doc);
void from5to6(QDomDocument& doc);
void from6to7(QDomDocument& doc);

void _rec_0to1(QDomElement doc);
void _rec_1to2(QDomElement doc);
void _rec_2to3(QDomElement doc);
void _rec_3to4(QDomNode doc);
void _rec_4to5(QDomNode doc);
void _rec_5to6(QDomNode doc);
QString get_version(QDomDocument const& doc);
}
};
}

#endif
