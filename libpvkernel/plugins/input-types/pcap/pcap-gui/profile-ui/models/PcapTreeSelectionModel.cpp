/*!
 * \file
 * \brief Manage the pcap protocols tree model.
 *
 * Manage the model used to bind the PcapTree structure to a Qt Model.
 *
 * \copyright (C) Philippe Saade - Justin Teliet 2014
 * \copyright (C) ESI Group INENDI 2017
 */

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
			return QVariant();
		}
	}
	return QVariant();
}

/**************************************************************************
 *
 * PcapTreeModel::data
 *
 *************************************************************************/
QVariant PcapTreeSelectionModel::data(const QModelIndex& index, int role) const
{
	if (not index.isValid())
		return QVariant();

	// Display protocol data.
	switch (role) {
	case Qt::DisplayRole: {
		JsonTreeItem* item = static_cast<JsonTreeItem*>(index.internalPointer());
		rapidjson::Value& value = item->value();
		// qreal ratio;

		switch (index.column()) {
		case 0: /* Name */
			return value["filter_name"].GetString();
		default:
			assert(false && "We only have 7 columns");
			return QVariant();
		}
		break;
	}

	case Qt::TextAlignmentRole:
		switch (index.column()) {
		case 0: /* Name */
			return Qt::AlignLeft;
		default:
			return QVariant();
		}
		break;
	}
	return QVariant();
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
