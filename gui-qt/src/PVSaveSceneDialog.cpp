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
	_scene(scene),
	_options(*options)
{
	setAcceptMode(QFileDialog::AcceptSave);
	setDefaultSuffix(PICVIZ_SCENE_ARCHIVE_EXT);
	setWindowTitle(tr("Save project..."));
	setNameFilters(QStringList() << PICVIZ_SCENE_ARCHIVE_FILTER << ALL_FILES_FILTER);

	QGridLayout* main_layout = (QGridLayout*) layout();

	_save_everything_checkbox = new QCheckBox(tr("Include formats and original files"));
	_save_everything_checkbox->setTristate(false);
	connect(_save_everything_checkbox, SIGNAL(stateChanged(int)), this, SLOT(include_files_Slot(int)));
	main_layout->addWidget(_save_everything_checkbox, 5, 1);

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
	tab_changed_Slot(0);

	connect(tabs, SIGNAL(currentChanged(int)), this, SLOT(tab_changed_Slot(int)));

	QVBoxLayout* layout = new QVBoxLayout();
	layout->addWidget(tabs);
	setLayout(layout);
}

void PVInspector::PVSaveSceneDialog::include_files_Slot(int state)
{
	if (state != Qt::PartiallyChecked) {
		_options.include_all_files(state == Qt::Checked);
	}
}

void PVInspector::PVSaveSceneDialog::tab_changed_Slot(int idx)
{
	if (idx == 0) {
		// AG: still part of the "include all" hack.
		// We're back ine the mai ntab, update the state of the check box according
		// to the options.
		disconnect(_save_everything_checkbox, SIGNAL(stateChanged(int)), this, SLOT(include_files_Slot(int)));
		_save_everything_checkbox->setCheckState((Qt::CheckState) _options.does_include_all_files());
		connect(_save_everything_checkbox, SIGNAL(stateChanged(int)), this, SLOT(include_files_Slot(int)));
	}
}
