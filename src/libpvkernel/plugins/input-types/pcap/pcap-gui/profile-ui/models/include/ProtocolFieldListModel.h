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

#ifndef PROTOCOLFIELDLISTMODEL_H
#define PROTOCOLFIELDLISTMODEL_H

#include <QAbstractItemModel>

#include "rapidjson/document.h"

/*!
 * Namespace that groups together the components of managing
 * the tree of protocols in a pcap file.
 */
namespace pvpcap
{

/**
 * Simple model to display list of fields for the selected protocol.
 */
class ProtocolFieldListModel : public QAbstractItemModel
{
	Q_OBJECT
  public:
	/**
	 * Keep a reference on the field to make it easy to visit.
	 */
	ProtocolFieldListModel(rapidjson::Value* fields, QObject* parent = 0);

	/**
	 * Simple forwarding of data as it is a raw data structure.
	 */
	QModelIndex index(int row, int column, const QModelIndex& parent) const override;

	/**
	 * We don't have parent.
	 */
	QModelIndex parent(const QModelIndex& index) const override;

	/**
	 * Get the row count
	 */
	int rowCount(const QModelIndex& parent) const override;

	/**
	 * Get the column count.
	 */
	int columnCount(const QModelIndex& parent) const override;

	/**
	 * Get the field information.
	 */
	QVariant data(const QModelIndex& index, int role) const override;

	/**
	 * Get the headers names.
	 */
	QVariant
	headerData(int section, Qt::Orientation orientation, int role = Qt::DisplayRole) const override;

	/**
	 * Get the flags of the given QIndexModel.
	 */
	Qt::ItemFlags flags(const QModelIndex& index) const override;

	/**
	 * Sets the role data for the item at index to value.
	 */
	bool setData(const QModelIndex& index, const QVariant& value, int role) override;

  Q_SIGNALS:
	/**
	 * Emit this signal to inform connected widget that selection have changed for field.
	 */
	void update_selection(rapidjson::Value&);

  public Q_SLOTS:
	/**
	 * Toogle selection for the given QModelIndex.
	 */
	void select(QModelIndex const&);

  protected:
	rapidjson::Value& _fields; //!< A rapidjson array of handled protocol fields.
};

} /* namespace pvpcap */

#endif /* PROTOCOLFIELDLISTMODEL_H */
