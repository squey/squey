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

#include <pvkernel/widgets/PVPresetsWidget.h>
#include <pvkernel/widgets/PVModdedIcon.h>
#include <QtCore/qobjectdefs.h>
#include <qabstractbutton.h>
#include <qflags.h>
#include <qicon.h>
#include <qlist.h>
#include <qsizepolicy.h>
#include <qstyle.h>
#include <QAction>
#include <QComboBox>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QMessageBox>
#include <QPushButton>
#include <QToolBar>

PVWidgets::PVPresetsWidget::PVPresetsWidget(const QString& title,
                                            QWidget* parent)
    : QWidget(parent)
{
	// groupBox
	_group_box = new QGroupBox(title);

	// ToolBar
	setObjectName("PVPresetsWidget");
	_toolbar = new QToolBar("Presets Widget ToolBar");
	_toolbar->setObjectName("QToolBar_of_PVPresetsWidget");
	_toolbar->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Fixed);
	_toolbar->setIconSize(QSize(22, 22));
	auto toolbar_layout = new QHBoxLayout();
	toolbar_layout->addStretch(1);
	toolbar_layout->addWidget(_toolbar);
	toolbar_layout->addStretch(1);

	// List
	_list = new __impl::PVPresetsListWidget(this);
	_list->setUniformItemSizes(true);

	// QAction toolbar/menu binding
	_loadAct = new QAction(PVModdedIcon("file-import"), tr("Load"), this);
	_toolbar->addAction(_loadAct);
	_list->addAction(_loadAct);
	_saveAct = new QAction(PVModdedIcon("file-export"), tr("Save"), this);
	_toolbar->addAction(_saveAct);
	_list->addAction(_saveAct);
	_renameAct = new QAction(QIcon(), tr("Rename"), this);
	_list->addAction(_renameAct);
	_removeAct = new QAction(QIcon(), tr("Remove"), this);
	_list->addAction(_removeAct);
	//_toolbar->addAction(_removeAct);
	_list->setContextMenuPolicy(Qt::ActionsContextMenu);

	// Connections
	connect(_loadAct, &QAction::triggered, this, &PVPresetsWidget::load_Slot);
	connect(_saveAct, &QAction::triggered, this, &PVPresetsWidget::save_Slot);
	connect(_renameAct, &QAction::triggered, this, &PVPresetsWidget::rename_Slot);
	connect(_removeAct, &QAction::triggered, this, &PVPresetsWidget::remove_Slot);
	connect(_list, &QListWidget::itemDoubleClicked, this, &PVPresetsWidget::load_Slot);
	connect(_list, &QListWidget::itemChanged, this, &PVPresetsWidget::item_changed_Slot);

	// Layout
	auto main_layout = new QVBoxLayout();
	main_layout->setContentsMargins(0, 0, 0, 0);
	auto group_box_layout = new QVBoxLayout();
	group_box_layout->addLayout(toolbar_layout);
	group_box_layout->addWidget(_list);
	_group_box->setLayout(group_box_layout);
	main_layout->addWidget(_group_box);
	setLayout(main_layout);

	update_actions_availability();
}

QVariant PVWidgets::PVPresetsWidget::get_preset_data(int index) const
{
	return _list->item(index)->data(Qt::UserRole);
}

QVariant PVWidgets::PVPresetsWidget::get_preset_data() const
{
	QListWidgetItem* item = _list->currentItem();
	if (!item) {
		return {};
	}
	return item->data(Qt::UserRole);
}

void PVWidgets::PVPresetsWidget::clear_presets()
{
	_list->clear();
}

QString PVWidgets::PVPresetsWidget::get_current_preset_name() const
{
	QListWidgetItem* item = _list->currentItem();
	if (!item) {
		return "";
	}
	return item->text();
}

int PVWidgets::PVPresetsWidget::get_preset_count() const
{
	return _list->count();
}

void PVWidgets::PVPresetsWidget::add_presets(const QStringList& presets)
{
	_list->addItems(presets);
	update_actions_availability();
}

void PVWidgets::PVPresetsWidget::add_preset(const QString& preset, const QVariant& userData)
{
	auto item = new QListWidgetItem(preset);
	item->setData(Qt::UserRole, userData);
	_list->addItem(item);
	update_actions_availability();
}

void PVWidgets::PVPresetsWidget::load_Slot()
{
	QListWidgetItem* item = _list->currentItem();
	if (!item) {
		return;
	}

	_list->blockSignals(true);
	for (int i = 0; i < _list->count(); i++) {
		QListWidgetItem* it = _list->item(i);
		it->setFlags(it->flags() & ~Qt::ItemIsEditable);
	}
	_list->blockSignals(false);

	_last_preset_loaded = item->text();
	Q_EMIT btn_load_clicked_Signal(_last_preset_loaded);
}

void PVWidgets::PVPresetsWidget::save_Slot()
{
	PVSavePresetAsDialog saveDlg(this);
	if (saveDlg.exec() == QDialog::Accepted) {
		QString preset = saveDlg._comboBox->currentText();
		if (is_preset_txt_new(preset)) {
			auto item = new QListWidgetItem(preset);
			_list->addItem(item);
			_list->sortItems();
			_list->setCurrentItem(item);
			Q_EMIT btn_new_clicked_Signal(preset);
			update_actions_availability();
		} else {
			Q_EMIT btn_save_clicked_Signal(preset);
		}
	}
}

void PVWidgets::PVPresetsWidget::rename_Slot()
{
	// backup item name in case of rename
	QListWidgetItem* item = _list->currentItem();
	if (!item) {
		return;
	}
	_old_preset_name = item->text();

	_list->blockSignals(true);
	item->setFlags(item->flags() | Qt::ItemIsEditable);
	_list->blockSignals(false);
	_list->editItem(item);
}

void PVWidgets::PVPresetsWidget::item_changed_Slot(QListWidgetItem* item)
{
	_list->blockSignals(true);
	item->setFlags(item->flags() & ~Qt::ItemIsEditable);

	QString new_preset_name = item->text();

	// Invalid preset name
	if (new_preset_name.isEmpty()) {
		auto* box = new QMessageBox(QMessageBox::Critical, tr("Invalid preset"),
		                                   tr("Invalid preset name"), QMessageBox::Ok, this);
		box->exec();
		new_preset_name = _old_preset_name;
		item->setText(new_preset_name);
	}
	// Already existing preset
	else if (_list->findItems(new_preset_name, Qt::MatchFixedString).count() > 1) {
		auto* box =
		    new QMessageBox(QMessageBox::Question, tr("Existing preset"),
		                    tr("The preset \"%1\" already exists. Replace?").arg(item->text()),
		                    QMessageBox::Yes | QMessageBox::No, this);
		if (box->exec() == QMessageBox::Yes) {
			delete _list->takeItem(_list->currentRow());
			update_actions_availability();
		} else {
			new_preset_name = _old_preset_name;
			item->setText(new_preset_name);
		}
	}

	if (_old_preset_name != new_preset_name) {
		Q_EMIT preset_renamed_Signal(_old_preset_name, new_preset_name);
	}

	_old_preset_name = "";
	_list->sortItems();
	_list->blockSignals(false);
}

void PVWidgets::PVPresetsWidget::update_actions_availability()
{
	bool enabled = _list->count();
	_loadAct->setEnabled(enabled);
	_renameAct->setEnabled(enabled);
	_removeAct->setEnabled(enabled);

	//	_list->removeAction(_loadAct);
	//	_list->removeAction(_saveAct);
	//	_list->removeAction(_renameAct);
	//	_list->removeAction(_removeAct);
	//
	//	if (enabled) {
	//		_list->addAction(_loadAct);
	//		_list->addAction(_saveAct);
	//		_list->addAction(_renameAct);
	//		_list->addAction(_removeAct);
	//	}
	//	else {
	//		_list->addAction(_saveAct);
	//	}
}

void PVWidgets::PVPresetsWidget::remove_Slot()
{
	QListWidgetItem* item = _list->currentItem();
	if (!item) {
		return;
	}
	QString preset = item->text();
	auto* box =
	    new QMessageBox(QMessageBox::Question, tr("Confirm delete"),
	                    tr("Are you sure you want to delete preset \"%1\"?").arg(preset),
	                    QMessageBox::Yes | QMessageBox::No, this);
	if (box->exec() == QMessageBox::Yes) {
		Q_EMIT btn_remove_clicked_Signal(preset);
		delete _list->takeItem(_list->currentRow());
		update_actions_availability();
	}
}

bool PVWidgets::PVPresetsWidget::is_preset_txt_new(const QString& str) const
{
	return _list->findItems(str, Qt::MatchFixedString).count() == 0;
}

PVWidgets::PVSavePresetAsDialog::PVSavePresetAsDialog(PVPresetsWidget* parent /*= 0*/)
    : QDialog(parent, Qt::Dialog), _parent(parent)
{
	setWindowModality(Qt::WindowModal);

	// Buttons
	_btn_save = new QPushButton(style()->standardIcon(QStyle::SP_DialogSaveButton), tr("Save"));
	auto* btn_cancel =
	    new QPushButton(style()->standardIcon(QStyle::SP_DialogCancelButton), tr("Cancel"));
	connect(_btn_save, &QAbstractButton::clicked, this, &PVSavePresetAsDialog::save_Slot);
	connect(btn_cancel, &QAbstractButton::clicked, this, &QDialog::reject);

	// Label
	auto* label = new QLabel(tr("Preset:"));

	// ComboBox
	_comboBox = new QComboBox();
	connect(_comboBox, &QComboBox::editTextChanged, this, &PVSavePresetAsDialog::text_changed_Slot);
	_comboBox->setEditable(true);
	if (!_parent->_last_preset_loaded.isEmpty()) {
		_comboBox->addItem(_parent->_last_preset_loaded);
		_comboBox->insertSeparator(1);
	}
	for (int i = 0; i < _parent->_list->count(); i++) {
		_comboBox->addItem(_parent->_list->item(i)->text());
	}
	QListWidgetItem* parent_selected_item = _parent->_list->currentItem();
	if (parent_selected_item /*&& parent_selected_item->text() != _parent->_last_preset_loaded*/) {
		_comboBox->setEditText(parent_selected_item->text());
	} else {
		_comboBox->setEditText("");
	}

	// Layouts
	auto h_layout1 = new QHBoxLayout();
	h_layout1->addWidget(label);
	h_layout1->addWidget(_comboBox, 1);
	auto h_layout2 = new QHBoxLayout();
	h_layout2->addWidget(_btn_save);
	h_layout2->addWidget(btn_cancel);
	auto main_layout = new QVBoxLayout();
	main_layout->addLayout(h_layout1);
	main_layout->addLayout(h_layout2);
	setLayout(main_layout);
}

void PVWidgets::PVSavePresetAsDialog::save_Slot()
{
	if (_comboBox->currentText().isEmpty()) {
		QMessageBox err(QMessageBox::Critical, tr("Invalid preset"),
		                tr("This preset name is invalid."), QMessageBox::Ok);
		err.exec();
	} else {
		accept();
	}
}

void PVWidgets::PVSavePresetAsDialog::text_changed_Slot(const QString& text)
{
	if (_comboBox->findText(text, Qt::MatchFixedString) != -1) {
		_btn_save->setText("Save");
		_btn_save->setIcon(style()->standardIcon(QStyle::SP_DialogSaveButton));
	} else {
		_btn_save->setText("New");
		_btn_save->setIcon(style()->standardIcon(QStyle::SP_FileIcon));
	}
}

void PVWidgets::PVPresetsWidget::select_preset(int idx)
{
	if (idx <= _list->count()) {
		_list->setCurrentRow(idx);
	}
}
