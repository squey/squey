/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2019
 */

#ifndef __RUSH_PVERFTREEMODEL_H__
#define __RUSH_PVERFTREEMODEL_H__

#include <QAbstractItemModel>

#include <memory>

#include <rapidjson/document.h>
#include <rapidjson/writer.h>       //
#include <rapidjson/stringbuffer.h> //

#include <pvlogger.h>

namespace PVRush
{

class PVERFTreeItem
{
  public:
	enum class EType { UNKOWN, NODE, STATES };

  public:
	explicit PVERFTreeItem(const QList<QVariant>& data, bool is_node, PVERFTreeItem* parent = 0);
	~PVERFTreeItem();

	void append_child(PVERFTreeItem* child);

	const PVERFTreeItem* child(int row) const;
	PVERFTreeItem* child(int row)
	{
		return const_cast<PVERFTreeItem*>(std::as_const(*this).child(row));
	}

	int child_count() const;
	int row() const;
	const PVERFTreeItem* parent() const;
	PVERFTreeItem* parent() { return const_cast<PVERFTreeItem*>(std::as_const(*this).parent()); }

	QVariant data(int column) const;

	void set_user_data(const QVariant& user_data) { _user_data = user_data; }
	const QVariant& user_data() const { return _user_data; }

	QString path() const;
	bool is_path(const QString& path) const;

	int column_count() const;

	Qt::CheckState state() const;
	void set_state(Qt::CheckState state);

	PVERFTreeItem* selected_child(int row);
	int selected_child_count() const;
	int selected_row() const;

	bool is_node() const { return _is_node; }
	EType type() const { return _type; }
	void set_type(EType type) { _type = type; }
	bool children_are_leafs() const { return child_count() > 0 and not child(0)->is_node(); }

  private:
	PVERFTreeItem* _parent = nullptr;
	std::vector<PVERFTreeItem*> _children;

	QList<QVariant> _data;
	QVariant _user_data;
	Qt::CheckState _state = Qt::CheckState::Unchecked;
	bool _is_node;
	EType _type = EType::UNKOWN;

#define STORE_INDEX 1
#if STORE_INDEX
	int _index = 0;
#endif
};

class PVERFTreeModel : public QAbstractItemModel
{
	Q_OBJECT;

  public:
	friend class PVERFTreeView;

  public:
	enum class ENodesType { ALL, SELECTED };

  private:
	template <typename NodeType>
	using visit_node_f = std::function<NodeType*(QModelIndex, NodeType*)>;

  public:
	PVERFTreeModel(const QString& path);

  public:
	bool load(const QString& path);
	QString path() const { return _path; }
	rapidjson::Document save(ENodesType nodes_type = ENodesType::SELECTED) const;

  public:
	QModelIndex
	index(int row, int column, const QModelIndex& parent = QModelIndex()) const override;

  Q_SIGNALS:
	void indexStateChanged(QModelIndex);

  protected:
	bool setData(const QModelIndex& index, const QVariant& value, int role) override;
	QVariant data(const QModelIndex& index, int role) const override;
	Qt::ItemFlags flags(const QModelIndex& index) const override;
	QVariant
	headerData(int section, Qt::Orientation orientation, int role = Qt::DisplayRole) const override;
	QModelIndex parent(const QModelIndex& index) const override;
	int rowCount(const QModelIndex& parent = QModelIndex()) const override;
	int columnCount(const QModelIndex& parent = QModelIndex()) const override;

  public:
	template <typename NodeType>
	void visit_nodes(ENodesType t, NodeType* root, const visit_node_f<NodeType>& f) const
	{
		QModelIndex index = this->index(0, 0);

		std::vector<std::pair<NodeType*, bool>> parents({std::make_pair(root, true)});

		visit_nodes_impl(t, f, index, this->rowCount(index) == 1, parents);
	}

  private:
	template <typename NodeType>
	void visit_nodes_impl(ENodesType t,
	                      const visit_node_f<NodeType>& f,
	                      QModelIndex index,
	                      bool is_last_child,
	                      std::vector<std::pair<NodeType*, bool>>& parents) const
	{
		PVERFTreeItem* item = static_cast<PVERFTreeItem*>(index.internalPointer());
		Qt::CheckState item_state = item->state();
		NodeType* child = nullptr;
		if (t == ENodesType::ALL or item_state != Qt::Unchecked) {
			child = f(index, parents.back().first);
			if (child) {
				parents.emplace_back(child, is_last_child);
			}
		}
		if (child) {
			const int child_count = item->child_count();
			int processed_child_count =
			    t == ENodesType::ALL ? child_count : item->selected_child_count();
			for (int i = 0; i < child_count; i++) {
				QModelIndex index_child = this->index(i, 0, index);
				PVERFTreeItem* child_item =
				    static_cast<PVERFTreeItem*>(index_child.internalPointer());
				Qt::CheckState child_item_state = child_item->state();
				if (t == ENodesType::ALL or child_item_state != Qt::Unchecked) {
					visit_nodes_impl(t, f, index_child, --processed_child_count == 0, parents);
				}
			}
		} else if (is_last_child) {
			while (parents.back().second) {
				parents.pop_back();
			}
			parents.pop_back();
		}
	}

  private:
	std::unique_ptr<PVERFTreeItem> _root_item;
	QString _path;
};

} // namespace PVRush

#endif // __RUSH_PVERFTREEMODEL_H__