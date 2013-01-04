/**
 * \file PVCustomQtRoles.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVGUIQT_PVCUSTOMQTROLES_H
#define PVGUIQT_PVCUSTOMQTROLES_H

#include <QMetaType>
Q_DECLARE_METATYPE(std::string)

namespace PVGuiQt {

namespace PVCustomQtRoles {

enum {
	Sort = Qt::UserRole,
	RoleSetSelectedItem
};

}

}

#endif
