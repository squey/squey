/**
 * \file PVImportSourceToProjectDlg.cpp
 *
 * Copyright (C) Picviz Labs 2012
 */

#include <picviz/PVRoot.h>
#include <picviz/PVScene.h>
#include <pvguiqt/PVImportSourceToProjectDlg.h>

#include <QDialogButtonBox>
#include <QBoxLayout>
#include <QLabel>
#include <QComboBox>


PVGuiQt::PVImportSourceToProjectDlg::PVImportSourceToProjectDlg(Picviz::PVRoot const& root, Picviz::PVScene const* sel_scene, QWidget* parent /* = 0 */) :
	QDialog(parent)
{
	setWindowTitle(tr("Select project"));

	QVBoxLayout* layout = new QVBoxLayout();

	QHBoxLayout* hbox_layout = new QHBoxLayout();
	QLabel* label = new QLabel(tr("Import source to data collection:"));

	_combo_box = new QComboBox();

	hbox_layout->addWidget(label);
	hbox_layout->addWidget(_combo_box);

	QDialogButtonBox* buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);

    connect(buttons, SIGNAL(accepted()), this, SLOT(accept()));
    connect(buttons, SIGNAL(rejected()), this, SLOT(reject()));

    layout->addLayout(hbox_layout);
	layout->addWidget(buttons);

	setLayout(layout);

	// Set combo box
	int cur_idx = 0;
	for (Picviz::PVScene_sp const& scene: root.get_children()) {
		QVariant var;
		var.setValue<void*>(scene.get());
		if (scene.get() == sel_scene) {
			cur_idx = _combo_box->count();
		}

		_combo_box->addItem(scene->get_name(), var);
	}
	_combo_box->setCurrentIndex(cur_idx);

	show();
}

Picviz::PVScene const* PVGuiQt::PVImportSourceToProjectDlg::get_selected_scene() const
{
	int sel_idx = _combo_box->currentIndex();
	Picviz::PVScene const* ret = (Picviz::PVScene const*) _combo_box->itemData(sel_idx).value<void*>();
	assert(ret);
	return ret;
}
