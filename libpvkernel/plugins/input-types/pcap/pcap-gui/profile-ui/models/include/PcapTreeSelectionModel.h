/*!
 * \file
 * \brief Manage the pcap protocols tree model.
 *
 * Manage the model used to bind the PcapTree structure to a Qt Model.
 *
 * \copyright (C) Philippe Saade - Justin Teliet 2014
 * \copyright (C) ESI Group INENDI 2017
 */
#ifndef PCAPTREESELECTIONMODEL_H
#define PCAPTREESELECTIONMODEL_H

#include "PcapTreeModel.h"

/*!
 * Namespace that groups together the components of managing
 * the tree of protocols in a pcap file.
 */
namespace pvpcap
{

/**
 * This is the model used to bind the PcapTree structure to a Qt Model.
 */
class PcapTreeSelectionModel : public PcapTreeModel
{
	Q_OBJECT

  public:
	/**
	 * Keep a reference on the _protocol_tree to make it easy to visit.
	 */
	explicit PcapTreeSelectionModel(rapidjson::Document* json_data, QObject* parent = 0)
	    : PcapTreeModel(json_data, false, parent)
	{
	}

	/**
	 * Number of elements in the `parent` level.
	 */
	// int rowCount(const QModelIndex& parent) const override;

	/**
	 * A tree always have 1 column only.
	 */
	int columnCount(const QModelIndex& parent) const override;

	/**
	 * Get the name of the node.
	 */
	// QVariant data(const QModelIndex& index, int role) const override;

	/**
	 * Get the name of the tree header.
	 */
	QVariant
	headerData(int section, Qt::Orientation orientation, int role = Qt::DisplayRole) const override;

	/**
	 * Get the name of the node.
	 */
	QVariant data(const QModelIndex& index, int role) const override;

	/**
	 * Reset data
	 */
	void reset();
};

} /* namespace pvpcap */

#endif /* PCAPTREESELECTIONMODEL_H */
