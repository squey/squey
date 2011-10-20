#ifndef PVSAVESCENEDIALOG_H
#define PVSAVESCENEDIALOG_H

#include <pvkernel/core/PVSerializeArchiveOptions_types.h>
#include <picviz/PVScene.h>

#include <QFileDialog>
#include <QWidget>

namespace PVInspector {

class PVSaveSceneDialog: public QFileDialog
{
public:
	PVSaveSceneDialog(Picviz::PVScene_p scene, PVCore::PVSerializeArchiveOptions_p options, QWidget* parent);

protected:
	Picviz::PVScene_p _scene;
};

}

#endif
