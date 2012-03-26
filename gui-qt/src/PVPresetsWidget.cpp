#include "PVPresetsWidget.h"

#include <pvkernel/core/PVLogger.h>
#include <PVPresetsWidget.h>

PVInspector::PVPresetsWidget::PVPresetsWidget(QWidget* parent, Qt::WindowFlags f) :
	QWidget(parent, f),
	_label(NULL),
	_combo(NULL),
	_btn_new(NULL),
	_btn_load(NULL),
	_btn_save(NULL),
	_btn_remove(NULL)
{
	// Create widgets
	_label = new QLabel(tr("Presets: "));
	_combo = new QComboBox();
	_combo->setEditable(true);
	_btn_new = new QPushButton(tr("New"));
	_btn_load = new QPushButton(tr("Load"));
	_btn_save = new QPushButton(tr("Save"));
	_btn_remove = new QPushButton(tr("Remove"));

	// Add to layout
	_layout = new QHBoxLayout();
	_layout->addWidget(_label);
	_layout->addWidget(_combo);
	_layout->addWidget(_btn_new);
	_layout->addWidget(_btn_load);
	_layout->addWidget(_btn_save);
	_layout->addWidget(_btn_remove);
	setLayout(_layout);

	// Connect to slots
	connect(_btn_new, SIGNAL(clicked()), this, SLOT(new_Slot()));
	connect(_btn_load, SIGNAL(clicked()), this, SLOT(load_Slot()));
	connect(_btn_save, SIGNAL(clicked()), this, SLOT(save_Slot()));
	connect(_btn_remove, SIGNAL(clicked()), this, SLOT(remove_Slot()));
	connect(_combo, SIGNAL(editTextChanged(const QString&)), this, SLOT(text_changed_Slot(const QString&)));

	// Update buttons state
	text_changed_Slot(_combo->currentText());
}

QVariant PVInspector::PVPresetsWidget::get_preset_data(int index)
{
	return _combo->itemData(index);
}

QVariant PVInspector::PVPresetsWidget::get_preset_data()
{
	return _combo->itemData(_combo->currentIndex());
}

void PVInspector::PVPresetsWidget::clear_presets()
{
	_combo->clear();
}

QString PVInspector::PVPresetsWidget::get_current_preset_name()
{
	return _combo->currentText();
}

int PVInspector::PVPresetsWidget::get_preset_count()
{
	return _combo->count();
}

void PVInspector::PVPresetsWidget::add_presets(const QStringList& presets)
{
	_combo->addItems(presets);
}

void PVInspector::PVPresetsWidget::add_preset(const QString& preset, const QVariant& userData)
{
	_combo->addItem(preset, userData);
}

void PVInspector::PVPresetsWidget::new_Slot()
{
	QString preset = _combo->currentText();

	emit btn_new_clicked_Signal(preset);

	_combo->addItem(preset);
	text_changed_Slot(preset);
}

void PVInspector::PVPresetsWidget::load_Slot()
{
	emit btn_load_clicked_Signal(_combo->currentText());
}

void PVInspector::PVPresetsWidget::save_Slot()
{
	emit btn_save_clicked_Signal(_combo->currentText());
}

void PVInspector::PVPresetsWidget::remove_Slot()
{
	QString preset = _combo->currentText();

	emit btn_remove_clicked_Signal(preset);

	_combo->removeItem(_combo->currentIndex());
}

void PVInspector::PVPresetsWidget::text_changed_Slot(const QString& text)
{
	if (_combo->currentText().isEmpty()) {
		_btn_new->setEnabled(false);
		_btn_save->setEnabled(false);
		_btn_load->setEnabled(false);
		_btn_remove->setEnabled(false);
		return;
	}

	bool preset_new = is_preset_txt_new();
	_btn_new->setEnabled(preset_new);
	_btn_save->setEnabled(!preset_new);
	_btn_load->setEnabled(!preset_new);
	_btn_remove->setEnabled(!preset_new);
}

bool PVInspector::PVPresetsWidget::is_preset_txt_new()
{
	return _combo->findText(_combo->currentText(), Qt::MatchFixedString) == -1;
}




