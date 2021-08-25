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

#include <pvkernel/widgets/PVArgumentListModel.h>
#include <QStandardItemModel>

/******************************************************************************
 *
 * PVWidgets::PVArgumentListModel::PVArgumentListModel
 *
 *****************************************************************************/
PVWidgets::PVArgumentListModel::PVArgumentListModel(QObject* parent)
    : QAbstractTableModel(parent), _args(nullptr)
{
}

/******************************************************************************
 *
 * PVWidgets::PVArgumentListModel::PVArgumentListModel
 *
 *****************************************************************************/
PVWidgets::PVArgumentListModel::PVArgumentListModel(PVCore::PVArgumentList& args, QObject* parent)
    : QAbstractTableModel(parent), _args(&args)
{
}

/******************************************************************************
 *
 * PVWidgets::PVArgumentListModel::columnCount
 *
 *****************************************************************************/
int PVWidgets::PVArgumentListModel::columnCount(const QModelIndex& parent) const
{
	// Same as above
	if (_args == nullptr || parent.isValid())
		return 0;

	return 1;
}

/******************************************************************************
 *
 * PVWidgets::PVArgumentListModel::data
 *
 *****************************************************************************/
QVariant PVWidgets::PVArgumentListModel::data(const QModelIndex& index, int role) const
{
	// We check if we have no args, and then restrict to the cases of Qt::DisplayRole and
	// Qt::EditRole
	if (_args == nullptr || (role != Qt::DisplayRole && role != Qt::EditRole))
		return QVariant();

	// We get an iterator for the Arguments
	auto it = _args->begin();
	// We jump to the row given by the Index
	std::advance(it, index.row());

	// We return the value of tha argument at that position
	return it->value();
}

/******************************************************************************
 *
 * PVWidgets::PVArgumentListModel::flags
 *
 *****************************************************************************/
Qt::ItemFlags PVWidgets::PVArgumentListModel::flags(const QModelIndex& index) const
{
	// nothing to say if we have no Arguments
	if (_args == nullptr) {
		return QAbstractTableModel::flags(index);
	}

	// We prepare an empty ItemFlags
	Qt::ItemFlags ret;

	// We set the flags in case we are in the first column
	if (index.column() == 0) {
		ret |= Qt::ItemIsEnabled | Qt::ItemIsEditable;
	}

	return ret;
}

/******************************************************************************
 *
 * PVWidgets::PVArgumentListModel::headerData
 *
 *****************************************************************************/
QVariant
PVWidgets::PVArgumentListModel::headerData(int section, Qt::Orientation orientation, int role) const
{
	return QAbstractTableModel::headerData(section, orientation, role);
}

/******************************************************************************
 *
 * PVWidgets::PVArgumentListModel::rowCount
 *
 *****************************************************************************/
int PVWidgets::PVArgumentListModel::rowCount(const QModelIndex& parent) const
{
	// Cf. QAbstractTableModel's documentation. This is for a table view.
	if (_args == nullptr || parent.isValid())
		return 0;

	return _args->size();
}

/******************************************************************************
 *
 * PVWidgets::PVArgumentListModel::set_args
 *
 *****************************************************************************/
void PVWidgets::PVArgumentListModel::set_args(PVCore::PVArgumentList& args)
{
	beginResetModel();
	_args = &args;
	endResetModel();
}

/******************************************************************************
 *
 * PVWidgets::PVArgumentListModel::setData
 *
 *****************************************************************************/
bool PVWidgets::PVArgumentListModel::setData(const QModelIndex& index,
                                             const QVariant& value,
                                             int role)
{
	if (_args == nullptr || index.column() != 0 || role != Qt::EditRole)
		return false;

	auto it = _args->begin();
	std::advance(it, index.row());
	if (it == _args->end())
		return false; // Should never happen !

	it->value() = value;

	Q_EMIT dataChanged(index, index);

	return true;
}
