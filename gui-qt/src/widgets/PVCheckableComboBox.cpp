#include "PVCheckableComboBox.h"

#include <QHBoxLayout>

PVInspector::PVCheckableComboBox::PVCheckableComboBox(QString name, QWidget *parent): QWidget(parent)
{
	_checked = true;	// Default is checked

	QHBoxLayout *layout = new QHBoxLayout;
	checkbox = new QCheckBox;
	checkbox->setCheckState(Qt::Checked);
	layout->addWidget(checkbox);
	label = new QLabel(name);
	layout->addWidget(label);
	combobox = new QComboBox;
	layout->addWidget(combobox);

	setLayout(layout);

	connect(checkbox, SIGNAL(stateChanged(int)), this, SLOT(checkStateChanged_Slot(int)));
}

void PVInspector::PVCheckableComboBox::setChecked(bool checked)
{
	_checked = checked;

	if (checked) {
		checkbox->setCheckState(Qt::Checked);
	} else {
		checkbox->setCheckState(Qt::Unchecked);
	}
}

void PVInspector::PVCheckableComboBox::addItems(QStringList items)
{
	combobox->addItems(items);
}

void PVInspector::PVCheckableComboBox::checkStateChanged_Slot(int state) 
{
	// works because Qt::Unchecked = 0 and Qt::Checked = 2

	if (!state) {
		_checked = false;
	} else {
		_checked = true;
	}

	combobox->setEnabled(state);
	label->setEnabled(state);
}

