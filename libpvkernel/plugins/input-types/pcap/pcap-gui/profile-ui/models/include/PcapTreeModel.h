/*!
 * \file
 * \brief Manage the pcap protocols tree model.
 *
 * Manage the model used to bind the PcapTree structure to a Qt Model.
 *
 * \copyright (C) Philippe Saade - Justin Teliet 2014
 * \copyright (C) ESI Group INENDI 2017
 */
#ifndef PCAPTREEMODEL_H
#define PCAPTREEMODEL_H

#include <QAbstractItemModel>

#include "rapidjson/document.h"

/*!
 * Namespace that groups together the components of managing
 * the tree of protocols in a pcap file.
 */
namespace pvpcap
{

/**
 * Help for manage Json tree in a  QAbstractItemModel
 */
class JsonTreeItem
{
  public:
	enum class CHILDREN_SELECTION_STATE { UNSELECTED, PARTIALY_SELECTED, TOTALY_SELECTED, UNKOWN };

  public:
	explicit JsonTreeItem(rapidjson::Value* value, JsonTreeItem* parent = 0);
	explicit JsonTreeItem(JsonTreeItem* parent = 0);
	~JsonTreeItem();

	void append_child(JsonTreeItem* child);

	JsonTreeItem* child(int row);
	int child_count() const;
	int row() const;
	JsonTreeItem* parent();

	rapidjson::Value& value() const;

	// selection object
	/**
	 * True if one field of the protocol is selected
	 * @return true or false
	 */
	CHILDREN_SELECTION_STATE selection_state() const;

	void select_all_children();
	void unselect_children();

	JsonTreeItem* selected_child(int row);
	int selected_child_count() const;
	int selected_row() const;

	static JsonTreeItem* load(rapidjson::Value* value, JsonTreeItem* parent = 0);

  private:
	rapidjson::Value* _value;
	JsonTreeItem* _parent;
	QList<JsonTreeItem*> _children;
};

/**
 * This is the model used to bind the PcapTree structure to a Qt Model.
 */
class PcapTreeModel : public QAbstractItemModel
{
	Q_OBJECT

  public:
	/**
	 * Keep a reference on the _protocol_tree to make it easy to visit.
	 */
	explicit PcapTreeModel(rapidjson::Document* json_data,
	                       bool with_selection = false,
	                       QObject* parent = 0)
	    : QAbstractItemModel(parent)
	    , _root_item(new JsonTreeItem())
	    , _json_data(json_data)
	    , _with_selection(with_selection)
	{
	}

	~PcapTreeModel() { delete _root_item; }

	/**
	 * Get a ModelIndex from a position in the tree.
	 */
	QModelIndex index(int row, int column, const QModelIndex& parent) const override;

	/**
	 * Get the parent of the given QIndexModel.
	 */
	QModelIndex parent(const QModelIndex& index) const override;

	/**
	 * Number of elements in the `parent` level.
	 */
	int rowCount(const QModelIndex& parent) const override;

	/**
	 * A tree always have 1 column only.
	 */
	int columnCount(const QModelIndex& parent) const override;

	/**
	 * Set the checkbox state
	 */
	bool setData(const QModelIndex& index, const QVariant& value, int role) override;

	/**
	 * Get the name of the node.
	 */
	QVariant data(const QModelIndex& index, int role) const override;

	/**
	 * Get the name of the tree header.
	 */
	QVariant
	headerData(int section, Qt::Orientation orientation, int role = Qt::DisplayRole) const override;

	/**
	 * Get the flags of the given QIndexModel.
	 */
	Qt::ItemFlags flags(const QModelIndex& index) const override;

	/**
	 * Load json data from pcap file
	 * @param filename pcap file
	 * @return true if it was loaded
	 */
	bool load(QString filename, bool& canceled);

	/**
	 * Get internal json data
	 * @return pointeur to internal rapidjson::Document
	 */
	rapidjson::Document* json_data() { return _json_data; }

	void reset();

  protected:
	JsonTreeItem* _root_item;        //!< point to the root node.
	rapidjson::Document* _json_data; //!< store profile JSON document.
	bool _with_selection;            //!< True to display only selected object, false otherwise
};

} /* namespace pvpcap */

#endif /* PCAPTREEMODEL_H */
