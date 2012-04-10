#include <pvkernel/core/PVLogger.h>
#include <pvkernel/widgets/PVPresetsWidget.h>

PVWidgets::PVPresetsWidget::PVPresetsWidget(const QString& title, QWidget* parent, Qt::WindowFlags f) :
	QWidget(parent, f),
	_group_box(NULL),
	_list(NULL),
	_btn_load(NULL),
	_btn_save(NULL),
	_btn_rename(NULL),
	_btn_remove(NULL)
{
	// Widgets
	_list = new QListWidget();
	_list->setMaximumHeight(70); // FIXME: Use SizePolicy instead

	//QString style = "QPushButton { background-color: white; border-width:1px; border-style: solid; border-color:#66006e;}";

	_btn_load = new QPushButton(tr("Load"));
	//_btn_load = new QPushButton(QIcon(":/load-preset"), "");
	//_btn_load->setMaximumSize(22, 22);
	//_btn_load->setStyleSheet(style);
	//_btn_load->setToolTip(tr("Load"));

	_btn_save = new QPushButton(tr("Save"));
	//_btn_save = new QPushButton(QIcon(":/save-preset"), "");
	//_btn_save->setMaximumSize(22, 22);
	//_btn_save->setStyleSheet(style);
	//_btn_save->setToolTip(tr("Save"));

	_btn_rename = new QPushButton(tr("Rename"));
	//_btn_rename = new QPushButton(QIcon(":/rename-preset"), "");
	//_btn_rename->setMaximumSize(22, 22);
	//_btn_rename->setStyleSheet(style);
	//_btn_rename->setToolTip(tr("Rename"));

	_btn_remove = new QPushButton(tr("Remove"));
	//_btn_remove = new QPushButton(QIcon(":/remove-preset"), "");
	//_btn_remove->setMaximumSize(22, 22);
	//_btn_remove->setStyleSheet(style);
	//_btn_remove->setToolTip(tr("Remove"));

	_group_box = new QGroupBox(title);

	// Layouts
	QVBoxLayout* main_layout = new QVBoxLayout();
	QVBoxLayout* v_layout = new QVBoxLayout();
	QHBoxLayout* h_layout = new QHBoxLayout();
	h_layout->addWidget(_btn_load);
	h_layout->addWidget(_btn_save);
	h_layout->addWidget(_btn_rename);
	h_layout->addWidget(_btn_remove);
//	h_layout->addStretch(1);
//	h_layout->setSpacing(1);
	v_layout->addWidget(_list);
	v_layout->addLayout(h_layout);
	_group_box->setLayout(v_layout);
	main_layout->addWidget(_group_box);
	setLayout(main_layout);

	// Connection
	connect(_list, SIGNAL(itemDoubleClicked(QListWidgetItem*)), this, SLOT(load_Slot()));
	connect(_btn_load, SIGNAL(clicked()), this, SLOT(load_Slot()));
	connect(_btn_save, SIGNAL(clicked()), this, SLOT(save_Slot()));
	connect(_btn_rename, SIGNAL(clicked()), this, SLOT(rename_Slot()));
	connect(_btn_remove, SIGNAL(clicked()), this, SLOT(remove_Slot()));
	connect(_list, SIGNAL(itemChanged(QListWidgetItem*)), this, SLOT(item_changed_Slot(QListWidgetItem*)));

	update_button_status();
}

QVariant PVWidgets::PVPresetsWidget::get_preset_data(int index) const
{
	return _list->item(index)->data(Qt::UserRole);
}

QVariant PVWidgets::PVPresetsWidget::get_preset_data() const
{
	return _list->currentItem()->data(Qt::UserRole);
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
	update_button_status();
}

void PVWidgets::PVPresetsWidget::add_preset(const QString& preset, const QVariant& userData)
{
	QListWidgetItem* item = new QListWidgetItem(preset);
	item->setData(Qt::UserRole, userData);
	_list->addItem(item);
	update_button_status();
}

void PVWidgets::PVPresetsWidget::load_Slot()
{
	QListWidgetItem* item = _list->currentItem();
	if (!item) {
		return;
	}

	_last_preset_loaded = item->text();
	emit btn_load_clicked_Signal(_last_preset_loaded);
}

void PVWidgets::PVPresetsWidget::save_Slot()
{
	PVSavePresetAsDialog saveDlg(this);
	if (saveDlg.exec() == QDialog::Accepted) {
		QString preset = saveDlg._comboBox->currentText();
		if (is_preset_txt_new(preset)) {
			QListWidgetItem* item = new QListWidgetItem(preset);
			_list->addItem(item);
			_list->sortItems();
			_list->setCurrentItem(item);
			emit btn_new_clicked_Signal(preset);
		}
		else {
			emit btn_save_clicked_Signal(preset);
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
	item->setFlags(item->flags ()|Qt::ItemIsEditable);
	_list->editItem(item);
	_list->blockSignals(false);
}

void PVWidgets::PVPresetsWidget::item_changed_Slot(QListWidgetItem* item)
{
	_list->blockSignals(true);
	item->setFlags(item->flags() &~ Qt::ItemIsEditable);

	QString new_preset_name = item->text();

	// Invalid preset name
	if (new_preset_name.isEmpty()) {
		QMessageBox* box = new QMessageBox(QMessageBox::Critical, tr("Invalid preset"), tr("Invalid preset name"), QMessageBox::Ok, this);
		box->exec();
		new_preset_name = _old_preset_name;
		item->setText(new_preset_name);
	}
	// Already existing preset
	else if(_list->findItems(new_preset_name, Qt::MatchFixedString).count() > 1) {
		QMessageBox* box = new QMessageBox(QMessageBox::Question, tr("Existing preset"), tr("The preset \"%1\" already exists. Replace?").arg(item->text()), QMessageBox::Yes | QMessageBox::No, this);
		if (box->exec() == QMessageBox::Yes) {
			delete _list->takeItem(_list->currentRow());
			update_button_status();
		}
		else {
			new_preset_name = _old_preset_name;
			item->setText(new_preset_name);
		}
	}

	if (_old_preset_name != new_preset_name) {
		emit preset_renamed_Signal(_old_preset_name, new_preset_name);
	}

	_old_preset_name = "";
	_list->sortItems();
	_list->blockSignals(false);
}

void PVWidgets::PVPresetsWidget::update_button_status(bool enabled)
{
	_btn_load->setEnabled(enabled);
	_btn_rename->setEnabled(enabled);
	_btn_remove->setEnabled(enabled);
}

void PVWidgets::PVPresetsWidget::update_button_status()
{
	bool enabled = _list->count();
	update_button_status(enabled);

}

void PVWidgets::PVPresetsWidget::remove_Slot()
{
	QListWidgetItem* item = _list->currentItem();
	if (!item) {
		return;
	}
	QString preset = item->text();
	QMessageBox* box = new QMessageBox(QMessageBox::Question, tr("Confirm delete"), tr("Are you sure you want to delete preset \"%1\"?").arg(preset), QMessageBox::Yes | QMessageBox::No, this);
	if (box->exec() == QMessageBox::Yes) {
		emit btn_remove_clicked_Signal(preset);
		delete _list->takeItem(_list->currentRow());
		update_button_status();
	}
}

bool PVWidgets::PVPresetsWidget::is_preset_txt_new(const QString& str) const
{
	return _list->findItems(str, Qt::MatchFixedString).count() == 0;
}

PVWidgets::PVSavePresetAsDialog::PVSavePresetAsDialog(PVPresetsWidget* parent /*= 0*/, Qt::WindowFlags f /*= 0*/) :
	QDialog(parent, Qt::Dialog),
	_parent(parent)
{
	setWindowModality(Qt::WindowModal);

	// Buttons
	_btn_save = new QPushButton(style()->standardIcon(QStyle::SP_DialogSaveButton), tr("Save"));
	QPushButton* btn_cancel = new QPushButton(style()->standardIcon(QStyle::SP_DialogCancelButton), tr("Cancel"));
	connect(_btn_save, SIGNAL(clicked()), this, SLOT(save_Slot()));
	connect(btn_cancel, SIGNAL(clicked()), this, SLOT(reject()));

	// Label
	QLabel* label = new QLabel(tr("Preset:"));

	// ComboBox
	_comboBox = new QComboBox();
	connect(_comboBox, SIGNAL(editTextChanged(const QString &)), this, SLOT(text_changed_Slot(const QString &)));
	_comboBox->setEditable(true);
	if (! _parent->_last_preset_loaded.isEmpty()) {
		_comboBox->addItem(_parent->_last_preset_loaded);
		_comboBox->insertSeparator(1);
	}
	for (int i = 0 ; i < _parent->_list->count() ; i++) {
		_comboBox->addItem(_parent->_list->item(i)->text());
	}
	QListWidgetItem* parent_selected_item = _parent->_list->currentItem();
	if (parent_selected_item /*&& parent_selected_item->text() != _parent->_last_preset_loaded*/) {
		_comboBox->setEditText(parent_selected_item->text());
	}
	else {
		_comboBox->setEditText("");
	}

	// Layouts
	QHBoxLayout* h_layout1 = new QHBoxLayout();
	h_layout1->addWidget(label);
	h_layout1->addWidget(_comboBox, 1);
	QHBoxLayout* h_layout2 = new QHBoxLayout();
	h_layout2->addWidget(_btn_save);
	h_layout2->addWidget(btn_cancel);
	QVBoxLayout* main_layout = new QVBoxLayout();
	main_layout->addLayout(h_layout1);
	main_layout->addLayout(h_layout2);
	setLayout(main_layout);
}

void PVWidgets::PVSavePresetAsDialog::save_Slot()
{
	if (_comboBox->currentText().isEmpty()) {
		QMessageBox err(QMessageBox::Critical, tr("Invalid preset"), tr("This preset name is invalid."), QMessageBox::Ok);
		err.exec();
	}
	else {
		accept();
		_parent->update_button_status(true);
	}
}

void PVWidgets::PVSavePresetAsDialog::text_changed_Slot(const QString& text)
{
	if (_comboBox->findText(text, Qt::MatchFixedString) != -1) {
		_btn_save->setText("Save");
		_btn_save->setIcon(style()->standardIcon(QStyle::SP_DialogSaveButton));
	}
	else {
		_btn_save->setText("New");
		_btn_save->setIcon(style()->standardIcon(QStyle::SP_FileIcon));
	}
}
