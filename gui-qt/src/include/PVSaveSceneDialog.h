#ifndef PVSAVESCENEDIALOG_H
#define PVSAVESCENEDIALOG_H

#include <pvkernel/core/PVSerializeArchiveOptions_types.h>
#include <picviz/PVScene.h>

#include <QCheckBox>
#include <QFileDialog>
#include <QWidget>

namespace PVInspector {

class PVSaveSceneDialog: public QFileDialog
{
	Q_OBJECT
public:
	PVSaveSceneDialog(Picviz::PVScene_p scene, PVCore::PVSerializeArchiveOptions_p options, QWidget* parent);

protected slots:
	void include_files_Slot(int state);
	void tab_changed_Slot(int idx);

protected:
	Picviz::PVScene_p _scene;
	PVCore::PVSerializeArchiveOptions& _options;
	QCheckBox* _save_everything_checkbox;
};

}

#endif
