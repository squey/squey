//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

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

	auto* layout = new QVBoxLayout();

	auto* hbox_layout = new QHBoxLayout();
	auto* label = new QLabel(tr("Import source to data collection:"));

	_combo_box = new QComboBox();

	hbox_layout->addWidget(label);
	hbox_layout->addWidget(_combo_box);

	auto* buttons =
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
