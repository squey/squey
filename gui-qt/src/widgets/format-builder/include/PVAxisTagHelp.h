#ifndef PVAXISTAGHELP_FILE_H
#define PVAXISTAGHELP_FILE_H

#include <pvkernel/core/general.h>
#include <picviz/PVLayerFilter.h>

#include "../../../ui_PVAxisTagHelp.h"

#include <QDialog>

namespace PVInspector {

class PVAxisTagHelp: public QDialog, private Ui::PVAxisTagHelp
{
	Q_OBJECT
public:
	PVAxisTagHelp(Picviz::PVLayerFilterTag sel_tag = Picviz::PVLayerFilterTag(), QWidget* parent = NULL);
};

}

#endif
