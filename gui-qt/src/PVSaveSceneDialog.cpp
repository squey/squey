#include "PVSaveSceneDialog.h"
#include "PVSerializeOptionsWidget.h"

#include <pvkernel/core/PVSerializeArchiveOptions.h>

#include <QFileSystemModel>
#include <QDir>
#include <QGridLayout>

PVInspector::PVSaveSceneDialog::PVSaveSceneDialog(Picviz::PVScene_p scene, PVCore::PVSerializeArchiveOptions_p options, QWidget* parent):
	QFileDialog(parent),
	_scene(scene)
{
	QGridLayout* main_layout = (QGridLayout*) layout();

	// Show the options
	PVSerializeOptionsWidget* widget_options = new PVSerializeOptionsWidget(options);
	main_layout->addWidget(widget_options, 6, 0, 1, -1);
}
