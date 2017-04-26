/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <inendi/PVRoot.h>
#include <inendi/PVScene.h>
#include <pvguiqt/PVImportSourceToProjectDlg.h>

#include <QDialogButtonBox>
#include <QBoxLayout>
#include <QLabel>
#include <QComboBox>

PVGuiQt::PVImportSourceToProjectDlg::PVImportSourceToProjectDlg(Inendi::PVRoot const& root,
                                                                Inendi::PVScene const* sel_scene,
                                                                QWidget* parent /* = 0 */)
    : QDialog(parent)
{
	setWindowTitle(tr("Select project"));

	QVBoxLayout* layout = new QVBoxLayout();

	QHBoxLayout* hbox_layout = new QHBoxLayout();
	QLabel* label = new QLabel(tr("Import source to data collection:"));

	_combo_box = new QComboBox();

	hbox_layout->addWidget(label);
	hbox_layout->addWidget(_combo_box);

	QDialogButtonBox* buttons =
	    new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);

	connect(buttons, &QDialogButtonBox::accepted, this, &QDialog::accept);
	connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);

	layout->addLayout(hbox_layout);
	layout->addWidget(buttons);

	setLayout(layout);

	// Set combo box
	int cur_idx = 0;
	for (auto const& scene : root.get_children()) {
		QVariant var;
		var.setValue<void*>(const_cast<Inendi::PVScene*>(scene));
		if (scene == sel_scene) {
			cur_idx = _combo_box->count();
		}

		_combo_box->addItem(QString::fromStdString(scene->get_name()), var);
	}
	_combo_box->setCurrentIndex(cur_idx);

	show();
}

Inendi::PVScene const* PVGuiQt::PVImportSourceToProjectDlg::get_selected_scene() const
{
	int sel_idx = _combo_box->currentIndex();
	Inendi::PVScene const* ret =
	    (Inendi::PVScene const*)_combo_box->itemData(sel_idx).value<void*>();
	assert(ret);
	return ret;
}
