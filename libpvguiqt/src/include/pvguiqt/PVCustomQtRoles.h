/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVGUIQT_PVCUSTOMQTROLES_H
#define PVGUIQT_PVCUSTOMQTROLES_H

#include <QMetaType>
Q_DECLARE_METATYPE(std::string)

namespace PVGuiQt
{

namespace PVCustomQtRoles
{

enum { Sort = Qt::UserRole, RoleSetSelectedItem, UnderlyingObject };
} // namespace PVCustomQtRoles
} // namespace PVGuiQt

#endif
