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

#include <PVXmlParamComboBox.h>
#include <QStandardItemModel>

/******************************************************************************
 *
 * PVInspector::PVXmlParamComboBox::PVXmlParamComboBox
 *
 *****************************************************************************/
PVInspector::PVXmlParamComboBox::PVXmlParamComboBox(QString name) : QComboBox()
{
	setObjectName(name);
	setModel(new PVComboBoxModel(_dis_elt));
}

/******************************************************************************
 *
 * PVInspector::PVXmlParamComboBox::~PVXmlParamComboBox
 *
 *****************************************************************************/
PVInspector::PVXmlParamComboBox::~PVXmlParamComboBox() = default;

/******************************************************************************
 *
 * QVariant PVInspector::PVXmlParamComboBox::val
 *
 *****************************************************************************/
QVariant PVInspector::PVXmlParamComboBox::val()
{
	// return the current selected item title.
	return this->currentText();
}

/******************************************************************************
 *
 * void PVInspector::PVXmlParamComboBox::select
 *
 *****************************************************************************/
void PVInspector::PVXmlParamComboBox::select(QString const& title)
{
	for (int i = 0; i < count(); i++) { // for each item...
		if (itemText(i) == title) {     //...if the title match...
			this->setCurrentIndex(i);   //...select it.
			break;
		}
	}
}

void PVInspector::PVXmlParamComboBox::add_disabled_string(QString const& str)
{
	_dis_elt.push_back(str);
}

void PVInspector::PVXmlParamComboBox::remove_disabled_string(QString const& str)
{
	int index = _dis_elt.indexOf(str);
	if (index != -1) {
		_dis_elt.removeAt(index);
	}
}

void PVInspector::PVXmlParamComboBox::clear_disabled_strings()
{
	_dis_elt.clear();
}

// PVComboBoxModel implementation

PVInspector::PVXmlParamComboBox::PVComboBoxModel::PVComboBoxModel(QStringList& dis_elt,
                                                                  QObject* parent)
    : QStandardItemModel(parent), _dis_elt(dis_elt)
{
}

Qt::ItemFlags
PVInspector::PVXmlParamComboBox::PVComboBoxModel::flags(const QModelIndex& index) const
{
	Qt::ItemFlags ret = QStandardItemModel::flags(index);
	if (is_disabled(index)) {
		// That item must be disabled (isn't selectable) !
		ret ^= Qt::ItemIsSelectable;
	}

	return ret;
}

QVariant PVInspector::PVXmlParamComboBox::PVComboBoxModel::data(const QModelIndex& index,
                                                                int role) const
{
	if (role == Qt::ForegroundRole && is_disabled(index)) {
		// If an item is disabled, it will be displayed in red.
		return QVariant(QBrush(QColor(255, 0, 0)));
	}

	return QStandardItemModel::data(index, role);
}

bool PVInspector::PVXmlParamComboBox::PVComboBoxModel::is_disabled(const QModelIndex& index) const
{
	return _dis_elt.indexOf(index.data().toString()) != -1;
}
