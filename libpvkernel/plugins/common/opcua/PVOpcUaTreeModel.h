/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2019
 */

#ifndef PVOPCUATREEMODEL_H
#define PVOPCUATREEMODEL_H

#include <QAbstractItemModel>
#include <QOpcUaNode>
#include <memory>

class QOpcUaClient;

namespace PVRush
{

class PVOpcUaTreeItem;

class PVOpcUaTreeModel : public QAbstractItemModel
{
  public:
	PVOpcUaTreeModel(QObject* parent = nullptr);

	void setOpcUaClient(QOpcUaClient*);
	QOpcUaClient* opcUaClient() const;

	QVariant data(const QModelIndex& index, int role) const override;
	QVariant headerData(int section, Qt::Orientation orientation, int role) const override;
	QModelIndex
	index(int row, int column, const QModelIndex& parent = QModelIndex()) const override;
	QModelIndex parent(const QModelIndex& index) const override;
	int rowCount(const QModelIndex& parent = QModelIndex()) const override;
	int columnCount(const QModelIndex& parent = QModelIndex()) const override;
	Qt::ItemFlags flags(const QModelIndex& index) const override;

  private:
	QOpcUaClient* m_client;
	std::unique_ptr<PVOpcUaTreeItem> m_root_item;

	friend class PVOpcUaTreeItem;
};

} // namespace PVRush

#endif
