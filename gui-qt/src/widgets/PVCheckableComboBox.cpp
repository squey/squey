#include "PVCheckableComboBox.h"

#include <QHBoxLayout>


PVInspector::PVCheckableComboBox::PVCheckableComboBox(QWidget *parent): QWidget(parent)
{
	_checked = true;	// Default is checked
	_current_index = 0;

	QHBoxLayout *layout = new QHBoxLayout;
	checkbox = new QCheckBox;
	checkbox->setCheckState(Qt::Checked);
	layout->addWidget(checkbox);
	label = new QLabel;
	layout->addWidget(label);
	combobox = new QComboBox;
	layout->addWidget(combobox);

	setLayout(layout);

	connect(checkbox, SIGNAL(stateChanged(int)), this, SLOT(checkStateChanged_Slot(int)));
	connect(combobox, SIGNAL(currentIndexChanged(int)), this, SLOT(comboIndexChanged_Slot(int)));

	setFocusPolicy(Qt::StrongFocus);
}

void PVInspector::PVCheckableComboBox::mouseReleaseEvent(QMouseEvent *ev)
{
	ev->accept();
	update();
}
// bool PVInspector::PVCheckableComboBox::eventFilter(QObject *o, QEvent *e)
// {
// 	// PVLOG_INFO("%s\n", __FUNCTION__);
// 	// combobox->editingFinished();
// 	return true;
// }

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

void PVInspector::PVCheckableComboBox::comboIndexChanged_Slot(int index)
{
	PVLOG_INFO("%s\n", __FUNCTION__);
	_current_index = index;
	update();
}

void PVInspector::PVCheckableComboBox::clear()
{
	combobox->clear();
}

void PVInspector::PVCheckableComboBox::setChecked(bool checked)
{
	PVLOG_INFO("WE SET CHECKED FOR OUR WIDGET\n");

	_checked = checked;

	if (checked) {
		checkbox->setCheckState(Qt::Checked);
	} else {
		checkbox->setCheckState(Qt::Unchecked);
	}
}

void PVInspector::PVCheckableComboBox::setText(QString text)
{
	label->setText(text);
}


