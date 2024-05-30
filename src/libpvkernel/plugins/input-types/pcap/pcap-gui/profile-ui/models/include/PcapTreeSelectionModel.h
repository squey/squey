/* * MIT License
 *
 * © ESI Group, 2015
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
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
