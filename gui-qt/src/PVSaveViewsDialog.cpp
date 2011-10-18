#include "PVSaveViewsDialog.h"
#include "PVSerializeOptionsWidget.h"

#include <pvkernel/core/PVSerializeArchiveOptions.h>
#include <picviz/PVView.h>

#include <QFileSystemModel>
#include <QDir>

PVInspector::PVSaveViewsDialog::PVSaveViewsDialog(QList<Picviz::PVView_p> const& views, QWidget* parent):
	QFileDialog(parent),
	_views(views)
{
	QGridLayout* main_layout = (QGridLayout*) layout();

	// Get options
	PVCore::PVSerializeArchiveOptions_p options(new PVCore::PVSerializeArchiveOptions(PICVIZ_ARCHIVES_VERSION));
	Picviz::PVSource_p src = views[0]->get_source_parent();
	options->get_root()->object("src", *src);

	// Show them
	PVSerializeOptionsWidget* widget_options = new PVSerializeOptionsWidget(options);
	main_layout->addWidget(widget_options, 6, 0, 1, -1);
}
