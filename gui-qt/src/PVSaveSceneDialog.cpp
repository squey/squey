#include "PVSaveSceneDialog.h"
#include "PVSerializeOptionsWidget.h"

#include <pvkernel/core/PVSerializeArchiveOptions.h>

#include <QFileSystemModel>
#include <QDir>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QPushButton>
#include <QSpacerItem>
#include <QTabWidget>
#include <QLabel>

PVInspector::PVSaveSceneDialog::PVSaveSceneDialog(Picviz::PVScene_p scene, PVCore::PVSerializeArchiveOptions_p options, QWidget* parent):
	QFileDialog(parent),
	_scene(scene)
{
	setAcceptMode(QFileDialog::AcceptSave);
	setDefaultSuffix(PICVIZ_SCENE_ARCHIVE_EXT);
	setWindowTitle(tr("Save project..."));
	setNameFilters(QStringList() << PICVIZ_SCENE_ARCHIVE_FILTER << ALL_FILES_FILTER);

	QGridLayout* main_layout = (QGridLayout*) layout();

	save_everything_checkbox = new QCheckBox(tr("Include formats and original files"));
	save_everything_checkbox->setChecked(false);
	main_layout->addWidget(save_everything_checkbox, 5, 1);

	QTabWidget* tabs = new QTabWidget();
	QWidget* org_w = new QWidget();
	org_w->setLayout(main_layout);
	tabs->addTab(org_w, tr("Project file"));

	// Show the options
	QVBoxLayout* options_layout = new QVBoxLayout();
	options_layout->addWidget(new QLabel(tr("You can choose which elements your project will contain.\nFor instance, source files can be included or not in the project.")));

	QHBoxLayout* options_h_l = new QHBoxLayout();
	PVSerializeOptionsWidget* widget_options = new PVSerializeOptionsWidget(options);
	options_h_l->addWidget(widget_options);
	QVBoxLayout* btn_layout = new QVBoxLayout();
	options_h_l->addLayout(btn_layout);

	QPushButton* expand_all_btn = new QPushButton(tr("Expand all"));
	QPushButton* collapse_all_btn = new QPushButton(tr("Collapse all"));
	connect(expand_all_btn, SIGNAL(clicked()), widget_options->get_view(), SLOT(expandAll()));
	connect(collapse_all_btn, SIGNAL(clicked()), widget_options->get_view(), SLOT(collapseAll()));
	btn_layout->addWidget(expand_all_btn);
	btn_layout->addWidget(collapse_all_btn);
	btn_layout->addItem(new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding));

	options_layout->addLayout(options_h_l);

	QWidget* options_widget = new QWidget();
	options_widget->setLayout(options_layout);
	tabs->addTab(options_widget, tr("Advanced Options"));

	QVBoxLayout* layout = new QVBoxLayout();
	layout->addWidget(tabs);
	setLayout(layout);

}

