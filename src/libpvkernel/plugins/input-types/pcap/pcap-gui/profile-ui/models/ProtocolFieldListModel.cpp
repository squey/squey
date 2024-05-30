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

#include <libpvpcap.h>
#include <libpvpcap/ws.h>
#include "include/ProtocolFieldListModel.h"
#include "iostream"

namespace pvpcap
{

/**************************************************************************
 *
 * ProtocolFieldListModel::ProtocolFieldListModel
 *
 *************************************************************************/
ProtocolFieldListModel::ProtocolFieldListModel(rapidjson::Value* fields, QObject* parent)
    : QAbstractItemModel(parent), _fields(*fields)
{
	assert(_fields.IsArray() && "The protocol fields is not a rapidjson array!");
}

/**************************************************************************
 *
 * ProtocolFieldListModel::index
 *
 *************************************************************************/

QModelIndex ProtocolFieldListModel::index(int row, int column, const QModelIndex&) const
{
	return createIndex(row, column, nullptr);
}

/**************************************************************************
 *
 * ProtocolFieldListModel::parent
 *
 *************************************************************************/
QModelIndex ProtocolFieldListModel::parent(const QModelIndex&) const
{
	return {};
}

/**************************************************************************
 *
 * ProtocolFieldListModel::columnCount
 *
 *************************************************************************/
int ProtocolFieldListModel::columnCount(const QModelIndex&) const
{
	return 5; /* Select, Name, Filter, Description, Type */
}

/**************************************************************************
 *
 * ProtocolFieldListModel::rowCount
 *
 *************************************************************************/
int ProtocolFieldListModel::rowCount(const QModelIndex&) const
{
	return _fields.Size();
}

/**************************************************************************
 *
 * ProtocolFieldListModel::data
 *
 *************************************************************************/
QVariant ProtocolFieldListModel::data(const QModelIndex& index, int role) const
{
	if (not index.isValid())
		return {};

	rapidjson::Value& value(_fields[index.row()]);

	if (role == Qt::DisplayRole) {
		switch (index.column()) {
		case 1:
			return value["name"].GetString();
		case 2:
			return value["filter_name"].GetString();
		case 3:
			return value["description"].GetString();
		case 4:
			return value["type"].GetString();
		}
	} else if (role == Qt::CheckStateRole) {
		if (index.column() == 0) // add a checkbox to cell(x,0)
			return value["select"].GetBool() ? Qt::Checked : Qt::Unchecked;
	}
	return {};
}

/**************************************************************************
 *
 * ProtocolFieldListModel::headerData
 *
 *************************************************************************/
QVariant
ProtocolFieldListModel::headerData(int section, Qt::Orientation orientation, int role) const
{
	if (role == Qt::DisplayRole && orientation == Qt::Horizontal) {
		switch (section) {
		case 1:
			return QString("Name");
		case 2:
			return QString("Filter");
		case 3:
			return QString("Description");
		case 4:
			return QString("Type");
		}
	}

	return {};
}

/**************************************************************************
 *
 * ProtocolFieldListModel::flags
 *
 *************************************************************************/
Qt::ItemFlags ProtocolFieldListModel::flags(const QModelIndex& index) const
{
	if (!index.isValid()) {
		return Qt::NoItemFlags;
	}

	const std::string& filter_name = _fields[index.row()]["filter_name"].GetString();
	if (pvpcap::ws_disabled_fields.find(filter_name) != pvpcap::ws_disabled_fields.end()) {
		return QAbstractItemModel::flags(index) & ~(Qt::ItemIsEnabled | Qt::ItemIsSelectable);
	}

	if (index.column() == 0) {
		return Qt::ItemIsEnabled | Qt::ItemIsSelectable;
	}

	return QAbstractItemModel::flags(index);
}

/**************************************************************************
 *
 * ProtocolFieldListModel::setData
 *
 *************************************************************************/
bool ProtocolFieldListModel::setData(const QModelIndex& index, const QVariant& value, int role)
{
	if (!index.isValid())
		return false;

	if (role == Qt::CheckStateRole) {

		rapidjson::Value& val(_fields[index.row()]);

		const std::string& filter_name = val["filter_name"].GetString();
		if (pvpcap::ws_disabled_fields.find(filter_name) != pvpcap::ws_disabled_fields.end()) {
			return false;
		}

		if ((Qt::CheckState)value.toInt() == Qt::Checked) {
			val["select"].SetBool(true);
		} else {
			val["select"].SetBool(false);
		}
		// Emit a new signal as index do not contains field information so we can't
		// know what have changed from the other signal.
		Q_EMIT update_selection(val);

		Q_EMIT dataChanged(index, index);

		return true;
	}
	return false;
}

/**************************************************************************
 *
 * ProtocolFieldListModel::select
 *
 *************************************************************************/
void ProtocolFieldListModel::select(QModelIndex const& index)
{
	Qt::CheckState state = static_cast<Qt::CheckState>(index.data(Qt::CheckStateRole).toInt());
	setData(index, (int)(state == Qt::Checked) ? Qt::Unchecked : Qt::Checked, Qt::CheckStateRole);
}

} // namespace pvpcap
