#ifndef PVINPUTTYPEMENUENTRIES_H
#define PVINPUTTYPEMENUENTRIES_H

#include <QMenu>
#include <QObject>

#include <pvkernel/rush/PVInputType.h>

namespace PVInspector {

class PVInputTypeMenuEntries
{
public:
	static void add_inputs_to_menu(QMenu* menu, QObject* parent, const char* slot);
	static PVRush::PVInputType_p input_type_from_action(QAction* action);
};

}

#endif
