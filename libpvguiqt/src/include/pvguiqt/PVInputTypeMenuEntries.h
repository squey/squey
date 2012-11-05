/**
 * \file PVInputTypeMenuEntries.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVINPUTTYPEMENUENTRIES_H
#define PVINPUTTYPEMENUENTRIES_H

#include <QMenu>
#include <QObject>
#include <QBoxLayout>

#include <pvkernel/rush/PVInputType.h>

namespace PVGuiQt {

class PVInputTypeMenuEntries
{
public:
	static void add_inputs_to_menu(QMenu* menu, QObject* parent, const char* slot);
	static void add_inputs_to_layout(QBoxLayout* layout, QObject* parent, const char* slot);
	static PVRush::PVInputType_p input_type_from_action(QAction* action);
};

}

#endif
