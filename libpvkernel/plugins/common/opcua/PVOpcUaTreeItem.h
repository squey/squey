/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2019
 */

#ifndef PVOPCUATREEITEM_H
#define PVOPCUATREEITEM_H

#include <QObject>
#include <QOpcUaNode>
#include <memory>

class QOpcUaRange;
class QOpcUaEUInformation;

namespace PVRush
{

class PVOpcUaTreeModel;

class PVOpcUaTreeItem : public QObject
{
  public:
	explicit PVOpcUaTreeItem(PVOpcUaTreeModel* model);
	PVOpcUaTreeItem(QOpcUaNode* node, PVOpcUaTreeModel* model, PVOpcUaTreeItem* parent);
	PVOpcUaTreeItem(QOpcUaNode* node,
	                PVOpcUaTreeModel* model,
	                const QOpcUaReferenceDescription& browsingData,
	                PVOpcUaTreeItem* parent);
	~PVOpcUaTreeItem();
	PVOpcUaTreeItem* child(int row);
	int childIndex(const PVOpcUaTreeItem* child) const;
	int childCount();
	int columnCount() const;
	QVariant data(int column);
	int row() const;
	PVOpcUaTreeItem* parentItem();
	void appendChild(PVOpcUaTreeItem* child);
	QPixmap icon(int column) const;
	bool hasChildNodeItem(const QString& nodeId) const;
	QVariant user_data(int column);
	bool has_history_access() const;

	void startBrowsing();
	void handleAttributes(QOpcUa::NodeAttributes attr);
	void browseFinished(QVector<QOpcUaReferenceDescription> children,
	                    QOpcUa::UaStatusCode statusCode);

  protected:
	std::unique_ptr<QOpcUaNode> m_opc_node;
	PVOpcUaTreeModel* m_model = nullptr;

  private:
	QString variantToString(const QVariant& value, const QString& typeNodeId = QString()) const;
	QString localizedTextToString(const QOpcUaLocalizedText& text) const;
	QString rangeToString(const QOpcUaRange& range) const;
	QString euInformationToString(const QOpcUaEUInformation& info) const;
	template <typename T>
	QString numberArrayToString(const QVector<T>& vec) const;

	bool m_attributes_ready = false;
	bool m_browse_started = false;
	QList<PVOpcUaTreeItem*> m_child_items;
	QSet<QString> m_child_node_ids;
	PVOpcUaTreeItem* m_parent_item = nullptr;

  private:
	QString m_node_browse_name;
	QString m_node_id;
	QString m_node_display_name;
	QOpcUa::NodeClass m_node_class = QOpcUa::NodeClass::Undefined;
};

template <typename T>
QString PVOpcUaTreeItem::numberArrayToString(const QVector<T>& vec) const
{
	QStringList list;
	for (auto it : vec)
		list.append(QString::number(it));

	return QLatin1String("[") + list.join(";") + QLatin1String("]");
}

} // namespace PVRush

#endif
