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

#include "PVOpcUaTreeItem.h"

#include "PVOpcUaTreeModel.h"

#include <QOpcUaArgument>
#include <QOpcUaAxisInformation>
#include <QOpcUaClient>
#include <QOpcUaComplexNumber>
#include <QOpcUaDoubleComplexNumber>
#include <QOpcUaEUInformation>
#include <QOpcUaExtensionObject>
#include <QOpcUaLocalizedText>
#include <QOpcUaQualifiedName>
#include <QOpcUaRange>
#include <QOpcUaXValue>
#include <QMetaEnum>
#include <QPixmap>
#include <QtOpcUa/qopcuanodeids.h>

namespace PVRush
{

const int numberOfDisplayColumns =
    7; // NodeId, Value, NodeClass, DataType, BrowseName, DisplayName, Description

PVOpcUaTreeItem::PVOpcUaTreeItem(PVOpcUaTreeModel* model) : QObject(nullptr), m_model(model) {}

PVOpcUaTreeItem::PVOpcUaTreeItem(QOpcUaNode* node, PVOpcUaTreeModel* model, PVOpcUaTreeItem* parent)
    : QObject(parent), m_opc_node(node), m_model(model), m_parent_item(parent)
{
	connect(m_opc_node.get(), &QOpcUaNode::attributeRead, this, &PVOpcUaTreeItem::handleAttributes);
	connect(m_opc_node.get(), &QOpcUaNode::browseFinished, this, &PVOpcUaTreeItem::browseFinished);

	if (!m_opc_node->readAttributes(
	        QOpcUa::NodeAttribute::Value | QOpcUa::NodeAttribute::NodeClass |
	        QOpcUa::NodeAttribute::Description | QOpcUa::NodeAttribute::DataType |
	        QOpcUa::NodeAttribute::BrowseName | QOpcUa::NodeAttribute::DisplayName |
	        QOpcUa::NodeAttribute::AccessLevel | QOpcUa::NodeAttribute::Historizing))
		qWarning() << "Reading attributes" << m_opc_node->nodeId() << "failed";
}

PVOpcUaTreeItem::PVOpcUaTreeItem(QOpcUaNode* node,
                                 PVOpcUaTreeModel* model,
                                 const QOpcUaReferenceDescription& browsingData,
                                 PVOpcUaTreeItem* parent)
    : PVOpcUaTreeItem(node, model, parent)
{
	m_node_browse_name = browsingData.browseName().name();
	m_node_class = browsingData.nodeClass();
	m_node_id = browsingData.targetNodeId().nodeId();
	m_node_display_name = browsingData.displayName().text();
}

PVOpcUaTreeItem::~PVOpcUaTreeItem()
{
	qDeleteAll(m_child_items);
}

PVOpcUaTreeItem* PVOpcUaTreeItem::child(int row)
{
	if (row >= m_child_items.size())
		qCritical() << "PVOpcUaTreeItem in row" << row << "does not exist.";
	return m_child_items[row];
}

int PVOpcUaTreeItem::childIndex(const PVOpcUaTreeItem* child) const
{
	return m_child_items.indexOf(const_cast<PVOpcUaTreeItem*>(child));
}

int PVOpcUaTreeItem::childCount()
{
	startBrowsing();
	return m_child_items.size();
}

int PVOpcUaTreeItem::columnCount() const
{
	return numberOfDisplayColumns;
}

QVariant PVOpcUaTreeItem::data(int column)
{
	if (column == 0) {
		return m_node_browse_name;
	} else if (column == 1) {
		if (!m_attributes_ready)
			return tr("Loading ...");

		const auto type = m_opc_node->attribute(QOpcUa::NodeAttribute::DataType).toString();
		const auto value = m_opc_node->attribute(QOpcUa::NodeAttribute::Value);

		return variantToString(value, type);
	} else if (column == 2) {
		QMetaEnum metaEnum = QMetaEnum::fromType<QOpcUa::NodeClass>();
		QString name = metaEnum.valueToKey((uint)m_node_class);
		return name + " (" + QString::number((uint)m_node_class) + ")";
	} else if (column == 3) {
		if (!m_attributes_ready)
			return tr("Loading ...");

		const QString typeId = m_opc_node->attribute(QOpcUa::NodeAttribute::DataType).toString();
		auto enumEntry = QOpcUa::namespace0IdFromNodeId(typeId);
		QString name;
		if (enumEntry == QOpcUa::NodeIds::Namespace0::Unknown)
			return typeId;
		return QOpcUa::namespace0IdName(enumEntry) + " (" + typeId + ")";
	} else if (column == 4) {
		return m_node_id;
	} else if (column == 5) {
		return m_node_display_name;
	} else if (column == 6) {
		if (!m_attributes_ready)
			return tr("Loading ...");

		return m_opc_node->attribute(QOpcUa::NodeAttribute::Description)
		    .value<QOpcUaLocalizedText>()
		    .text();
	}
	return QVariant();
}

bool PVOpcUaTreeItem::has_history_access() const
{
	const QString typeId = m_opc_node->attribute(QOpcUa::NodeAttribute::DataType).toString();
	switch (QOpcUa::namespace0IdFromNodeId(typeId)) {
		case QOpcUa::NodeIds::Namespace0::Boolean:
		case QOpcUa::NodeIds::Namespace0::SByte:
		case QOpcUa::NodeIds::Namespace0::Byte:
		case QOpcUa::NodeIds::Namespace0::Int16:
		case QOpcUa::NodeIds::Namespace0::UInt16:
		case QOpcUa::NodeIds::Namespace0::Int32:
		case QOpcUa::NodeIds::Namespace0::UInt32:
		case QOpcUa::NodeIds::Namespace0::Int64:
		case QOpcUa::NodeIds::Namespace0::UInt64:
		case QOpcUa::NodeIds::Namespace0::Float:
		case QOpcUa::NodeIds::Namespace0::Double:
		// case QOpcUa::NodeIds::Namespace0::String:
		case QOpcUa::NodeIds::Namespace0::DateTime:
			break;
		default: return false;
	}
	return m_attributes_ready and
	       (m_opc_node->attribute(QOpcUa::NodeAttribute::AccessLevel).toUInt() &
	        quint8(QOpcUa::AccessLevelBit::HistoryRead));
}

int PVOpcUaTreeItem::row() const
{
	if (!m_parent_item)
		return 0;
	return m_parent_item->childIndex(this);
}

PVOpcUaTreeItem* PVOpcUaTreeItem::parentItem()
{
	return m_parent_item;
}

void PVOpcUaTreeItem::appendChild(PVOpcUaTreeItem* child)
{
	if (!child)
		return;

	if (!hasChildNodeItem(child->m_node_id)) {
		m_child_items.append(child);
		m_child_node_ids.insert(child->m_node_id);
	} else {
		child->deleteLater();
	}
}

QPixmap PVOpcUaTreeItem::icon(int column) const
{
	if (column != 0 || !m_opc_node)
		return QPixmap();

	QColor c;

	switch (m_node_class) {
	case QOpcUa::NodeClass::Object:
		c = Qt::gray;
		break;
	case QOpcUa::NodeClass::Variable:
		if (has_history_access()) {
			c = Qt::darkGreen;
			break;
		}
		c = Qt::darkRed;
		break;
	case QOpcUa::NodeClass::Method:
		c = Qt::darkBlue;
		break;
	default:
		c = Qt::gray;
	}

	QPixmap p(10, 10);
	p.fill(c);
	return p;
}

bool PVOpcUaTreeItem::hasChildNodeItem(const QString& nodeId) const
{
	return m_child_node_ids.contains(nodeId);
}

void PVOpcUaTreeItem::startBrowsing()
{
	if (m_browse_started)
		return;

	if (!m_opc_node->browseChildren())
		qWarning() << "Browsing node" << m_opc_node->nodeId() << "failed";
	else
		m_browse_started = true;
}

void PVOpcUaTreeItem::handleAttributes(QOpcUa::NodeAttributes attr)
{
	if (attr & QOpcUa::NodeAttribute::NodeClass)
		m_node_class =
		    m_opc_node->attribute(QOpcUa::NodeAttribute::NodeClass).value<QOpcUa::NodeClass>();
	if (attr & QOpcUa::NodeAttribute::BrowseName)
		m_node_browse_name = m_opc_node->attribute(QOpcUa::NodeAttribute::BrowseName)
		                         .value<QOpcUaQualifiedName>()
		                         .name();
	if (attr & QOpcUa::NodeAttribute::DisplayName)
		m_node_display_name = m_opc_node->attribute(QOpcUa::NodeAttribute::DisplayName)
		                          .value<QOpcUaLocalizedText>()
		                          .text();

	m_attributes_ready = true;
	m_model->dataChanged(m_model->createIndex(row(), 0, this),
	                     m_model->createIndex(row(), numberOfDisplayColumns - 1, this));
}

void PVOpcUaTreeItem::browseFinished(QVector<QOpcUaReferenceDescription> children,
                                     QOpcUa::UaStatusCode statusCode)
{
	if (statusCode != QOpcUa::Good) {
		qWarning() << "Browsing node" << m_opc_node->nodeId() << "finally failed:" << statusCode;
		return;
	}

	auto index = m_model->createIndex(row(), 0, this);

	for (const auto& item : children) {
		if (hasChildNodeItem(item.targetNodeId().nodeId()))
			continue;

		auto node = m_model->opcUaClient()->node(item.targetNodeId());
		if (!node) {
			qWarning() << "Failed to instantiate node:" << item.targetNodeId().nodeId();
			continue;
		}

		m_model->beginInsertRows(index, m_child_items.size(), m_child_items.size() + 1);
		appendChild(new PVOpcUaTreeItem(node, m_model, item, this));
		m_model->endInsertRows();
	}

	m_model->dataChanged(m_model->createIndex(row(), 0, this),
	                     m_model->createIndex(row(), numberOfDisplayColumns - 1, this));
}

QVariant PVOpcUaTreeItem::user_data(int column)
{
	switch (column)
	{
	case 0:
		return m_node_id;
	case 3:
		if (!m_attributes_ready)
			return tr("Loading ...");
		return m_opc_node->attribute(QOpcUa::NodeAttribute::DataType).toString();
	default:
		return data(column);
	}
}

QString PVOpcUaTreeItem::variantToString(const QVariant& value, const QString& typeNodeId) const
{
	if (value.typeId() == QVariant::List) {
		const auto list = value.toList();
		QStringList concat;

		for (auto it : list)
			concat.append(variantToString(it, typeNodeId));

		return concat.join("\n");
	}

	if (typeNodeId == QLatin1String("ns=0;i=19")) { // StatusCode
		const char* name = QMetaEnum::fromType<QOpcUa::UaStatusCode>().valueToKey(value.toInt());
		if (!name)
			return QLatin1String("Unknown StatusCode");
		else
			return QString(name);
	}
	if (typeNodeId == QLatin1String("ns=0;i=2")) // Char
		return QString::number(value.toInt());
	else if (typeNodeId == QLatin1String("ns=0;i=3")) // SChar
		return QString::number(value.toUInt());
	else if (typeNodeId == QLatin1String("ns=0;i=4")) // Int16
		return QString::number(value.toInt());
	else if (typeNodeId == QLatin1String("ns=0;i=5")) // UInt16
		return QString::number(value.toUInt());
	else if (value.typeId() == QVariant::ByteArray)
		return QLatin1String("0x") + value.toByteArray().toHex();
	else if (value.typeId() == QVariant::DateTime)
		return value.toDateTime().toString(Qt::ISODate);
	else if (value.canConvert<QOpcUaQualifiedName>()) {
		const auto name = value.value<QOpcUaQualifiedName>();
		return QStringLiteral("[NamespaceIndex: %1, Name: \"%2\"]")
		    .arg(name.namespaceIndex())
		    .arg(name.name());
	} else if (value.canConvert<QOpcUaLocalizedText>()) {
		const auto text = value.value<QOpcUaLocalizedText>();
		return localizedTextToString(text);
	} else if (value.canConvert<QOpcUaRange>()) {
		const auto range = value.value<QOpcUaRange>();
		return rangeToString(range);
	} else if (value.canConvert<QOpcUaComplexNumber>()) {
		const auto complex = value.value<QOpcUaComplexNumber>();
		return QStringLiteral("[Real: %1, Imaginary: %2]")
		    .arg(complex.real())
		    .arg(complex.imaginary());
	} else if (value.canConvert<QOpcUaDoubleComplexNumber>()) {
		const auto complex = value.value<QOpcUaDoubleComplexNumber>();
		return QStringLiteral("[Real: %1, Imaginary: %2]")
		    .arg(complex.real())
		    .arg(complex.imaginary());
	} else if (value.canConvert<QOpcUaXValue>()) {
		const auto xv = value.value<QOpcUaXValue>();
		return QStringLiteral("[X: %1, Value: %2]").arg(xv.x()).arg(xv.value());
	} else if (value.canConvert<QOpcUaEUInformation>()) {
		const auto info = value.value<QOpcUaEUInformation>();
		return euInformationToString(info);
	} else if (value.canConvert<QOpcUaAxisInformation>()) {
		const auto info = value.value<QOpcUaAxisInformation>();
		return QStringLiteral(
		           "[EUInformation: %1, EURange: %2, Title: %3 , AxisScaleType: %4, AxisSteps: %5]")
		    .arg(euInformationToString(info.engineeringUnits()))
		    .arg(rangeToString(info.eURange()))
		    .arg(localizedTextToString(info.title()))
		    .arg(info.axisScaleType() == QOpcUa::AxisScale::Linear
		             ? "Linear"
		             : (info.axisScaleType() == QOpcUa::AxisScale::Ln) ? "Ln" : "Log")
		    .arg(numberArrayToString(info.axisSteps()));
	} else if (value.canConvert<QOpcUaExpandedNodeId>()) {
		const auto id = value.value<QOpcUaExpandedNodeId>();
		return QStringLiteral("[NodeId: \"%1\", ServerIndex: \"%2\", NamespaceUri: \"%3\"]")
		    .arg(id.nodeId())
		    .arg(id.serverIndex())
		    .arg(id.namespaceUri());
	} else if (value.canConvert<QOpcUaArgument>()) {
		const auto a = value.value<QOpcUaArgument>();

		return QStringLiteral("[Name: \"%1\", DataType: \"%2\", ValueRank: \"%3\", "
		                      "ArrayDimensions: %4, Description: %5]")
		    .arg(a.name())
		    .arg(a.dataTypeId())
		    .arg(a.valueRank())
		    .arg(numberArrayToString(a.arrayDimensions()))
		    .arg(localizedTextToString(a.description()));
	} else if (value.canConvert<QOpcUaExtensionObject>()) {
		const auto obj = value.value<QOpcUaExtensionObject>();
		return QStringLiteral("[TypeId: \"%1\", Encoding: %2, Body: 0x%3]")
		    .arg(obj.encodingTypeId())
		    .arg(obj.encoding() == QOpcUaExtensionObject::Encoding::NoBody
		             ? "NoBody"
		             : (obj.encoding() == QOpcUaExtensionObject::Encoding::ByteString ? "ByteString"
		                                                                              : "XML"))
		    .arg(obj.encodedBody().isEmpty() ? "0" : QString(obj.encodedBody().toHex()));
	}

	if (value.canConvert<QString>())
		return value.toString();

	return QString();
}

QString PVOpcUaTreeItem::localizedTextToString(const QOpcUaLocalizedText& text) const
{
	return QStringLiteral("[Locale: \"%1\", Text: \"%2\"]").arg(text.locale()).arg(text.text());
}

QString PVOpcUaTreeItem::rangeToString(const QOpcUaRange& range) const
{
	return QStringLiteral("[Low: %1, High: %2]").arg(range.low()).arg(range.high());
}

QString PVOpcUaTreeItem::euInformationToString(const QOpcUaEUInformation& info) const
{
	return QStringLiteral("[UnitId: %1, NamespaceUri: \"%2\", DisplayName: %3, Description: %4]")
	    .arg(info.unitId())
	    .arg(info.namespaceUri())
	    .arg(localizedTextToString(info.displayName()))
	    .arg(localizedTextToString(info.description()));
}

} // namespace PVRush
