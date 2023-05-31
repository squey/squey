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

#include <pvkernel/widgets/PVComboBox.h>

/******************************************************************************
 *
 * App::PVComboBox::PVComboBox
 *
 *****************************************************************************/
PVWidgets::PVComboBox::PVComboBox(QWidget* parent) : QComboBox(parent)
{
	setModel(new PVComboBoxModel(_dis_elt));
}

/******************************************************************************
 *
 * PVWidgets::PVComboBox::val
 *
 *****************************************************************************/
QString PVWidgets::PVComboBox::get_selected() const
{
	return currentText();
}

QVariant PVWidgets::PVComboBox::get_sel_userdata() const
{
	int idx = currentIndex();
	if (idx == -1) {
		return {};
	}
	return itemData(idx);
}

/******************************************************************************
 *
 * void PVWidgets::PVComboBox::select
 *
 *****************************************************************************/
bool PVWidgets::PVComboBox::select(QString const& title)
{
	int idx = findText(title);
	if (idx == -1) {
		return false;
	}

	setCurrentIndex(idx);
	return true;
}

bool PVWidgets::PVComboBox::select_userdata(QVariant const& data)
{
	int idx = findData(data);
	if (idx == -1) {
		return false;
	}

	setCurrentIndex(idx);
	return true;
}

void PVWidgets::PVComboBox::add_disabled_string(QString const& str)
{
	_dis_elt.push_back(str);
}

void PVWidgets::PVComboBox::remove_disabled_string(QString const& str)
{
	int index = _dis_elt.indexOf(str);
	if (index != -1) {
		_dis_elt.removeAt(index);
	}
}

void PVWidgets::PVComboBox::clear_disabled_strings()
{
	_dis_elt.clear();
}

// PVComboBoxModel implementation

PVWidgets::PVComboBox::PVComboBoxModel::PVComboBoxModel(QStringList& dis_elt, QObject* parent)
    : QStandardItemModel(parent), _dis_elt(dis_elt)
{
}

Qt::ItemFlags PVWidgets::PVComboBox::PVComboBoxModel::flags(const QModelIndex& index) const
{
	Qt::ItemFlags ret = QStandardItemModel::flags(index);
	if (is_disabled(index)) {
		// That item must be disabled (isn't selectable) !
		ret ^= Qt::ItemIsSelectable;
	}

	return ret;
}

QVariant PVWidgets::PVComboBox::PVComboBoxModel::data(const QModelIndex& index, int role) const
{
	if (role == Qt::ForegroundRole && is_disabled(index)) {
		// If an item is disabled, it will be displayed in red.
		return QVariant(QBrush(QColor(255, 0, 0)));
	}

	return QStandardItemModel::data(index, role);
}

bool PVWidgets::PVComboBox::PVComboBoxModel::is_disabled(const QModelIndex& index) const
{
	return _dis_elt.indexOf(index.data().toString()) != -1;
}
