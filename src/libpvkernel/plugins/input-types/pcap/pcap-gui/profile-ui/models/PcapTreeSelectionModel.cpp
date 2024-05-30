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
#include "include/PcapTreeSelectionModel.h"

namespace pvpcap
{

/**************************************************************************
 *
 * PcapTreeSelectionModel::columnCount
 *
 *************************************************************************/
int PcapTreeSelectionModel::columnCount(const QModelIndex& parent) const
{
	Q_UNUSED(parent);

	return 1; /* Name, Short name, Filter, Packets, % Packets, Bytes, % Bytes */
}

/**************************************************************************
 *
 * PcapTreeSelectionModel::headerData
 *
 *************************************************************************/
QVariant
PcapTreeSelectionModel::headerData(int section, Qt::Orientation orientation, int role) const
{
	if (orientation == Qt::Horizontal && role == Qt::DisplayRole) {
		switch (section) {
		case 0: /* Name */
			return "Name";

		default:
			assert(false && "We only have 7 columns");
			return {};
		}
	}
	return {};
}

/**************************************************************************
 *
 * PcapTreeModel::data
 *
 *************************************************************************/
QVariant PcapTreeSelectionModel::data(const QModelIndex& index, int role) const
{
	if (not index.isValid())
		return {};

	// Display protocol data.
	switch (role) {
	case Qt::DisplayRole: {
		auto* item = static_cast<JsonTreeItem*>(index.internalPointer());
		rapidjson::Value& value = item->value();
		// qreal ratio;

		switch (index.column()) {
		case 0: /* Name */
			return value["filter_name"].GetString();
		default:
			assert(false && "We only have 7 columns");
			return {};
		}
		break;
	}

	case Qt::TextAlignmentRole:
		switch (index.column()) {
		case 0: /* Name */
			return Qt::AlignLeft;
		default:
			return {};
		}
		break;
	}
	return {};
}

/*************************************************************************
 *
 * PcapTreeSelectionModel::reset
 *
 *************************************************************************/
void PcapTreeSelectionModel::reset()
{
	beginResetModel();
	// Fixme: comment ecraser l'arbre
	_root_item = JsonTreeItem::load(_json_data);
	endResetModel();
}

} // namespace pvpcap
