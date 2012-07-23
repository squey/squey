/**
 * \file PVComboBox.cpp
 *
 * Copyright (C) Picviz Labs 2011-2012
 */

#include <PVComboBox.h>


/******************************************************************************
 *
 * PVInspector::PVComboBox::PVComboBox
 *
 *****************************************************************************/
PVInspector::PVWidgetsHelpers::PVComboBox::PVComboBox(QWidget* parent):
	QComboBox(parent)
{
	setModel(new PVComboBoxModel(_dis_elt));
}

/******************************************************************************
 *
 * PVInspector::PVWidgetsHelpers::PVComboBox::val
 *
 *****************************************************************************/
QString PVInspector::PVWidgetsHelpers::PVComboBox::get_selected() const
{
    return currentText();
}

QVariant PVInspector::PVWidgetsHelpers::PVComboBox::get_sel_userdata() const
{
	int idx = currentIndex();
	if (idx == -1) {
		return QVariant();
	}
	return itemData(idx);
}

/******************************************************************************
 *
 * void PVInspector::PVWidgetsHelpers::PVComboBox::select
 *
 *****************************************************************************/
bool PVInspector::PVWidgetsHelpers::PVComboBox::select(QString const& title)
{
	int idx = findText(title);
	if (idx == -1) {
		return false;
	}

	setCurrentIndex(idx);
	return true;
}

bool PVInspector::PVWidgetsHelpers::PVComboBox::select_userdata(QVariant const& data)
{
	int idx = findData(data);
	if (idx == -1) {
		return false;
	}

	setCurrentIndex(idx);
	return true;
}

void PVInspector::PVWidgetsHelpers::PVComboBox::add_disabled_string(QString const& str)
{
	_dis_elt.push_back(str);
}

void PVInspector::PVWidgetsHelpers::PVComboBox::remove_disabled_string(QString const& str)
{
	int index = _dis_elt.indexOf(str);
	if (index != -1) {
		_dis_elt.removeAt(index);
	}
}

void PVInspector::PVWidgetsHelpers::PVComboBox::clear_disabled_strings()
{
	_dis_elt.clear();
}

// PVComboBoxModel implementation

PVInspector::PVWidgetsHelpers::PVComboBox::PVComboBoxModel::PVComboBoxModel(QStringList& dis_elt, QObject* parent):
	QStandardItemModel(parent),
	_dis_elt(dis_elt)
{
}

Qt::ItemFlags PVInspector::PVWidgetsHelpers::PVComboBox::PVComboBoxModel::flags(const QModelIndex& index) const
{
	Qt::ItemFlags ret = QStandardItemModel::flags(index);
	if (is_disabled(index)) {
		// That item must be disabled (isn't selectable) !
		ret ^= Qt::ItemIsSelectable;
	}

	return ret;
}

QVariant PVInspector::PVWidgetsHelpers::PVComboBox::PVComboBoxModel::data(const QModelIndex& index, int role) const
{
	if (role == Qt::ForegroundRole && is_disabled(index)) {
		// If an item is disabled, it will be displayed in red.
		return QVariant(QBrush(QColor(255,0,0)));
	}

	return QStandardItemModel::data(index, role);
}

bool PVInspector::PVWidgetsHelpers::PVComboBox::PVComboBoxModel::is_disabled(const QModelIndex& index) const
{
	return _dis_elt.indexOf(index.data().toString()) != -1;
}
