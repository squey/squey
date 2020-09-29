#include "PVERFTreeView.h"

PVRush::PVERFTreeView::PVERFTreeView(PVRush::PVERFTreeModel* model, QWidget* parent /*= nullptr*/)
    : QTreeView(parent), _model(model)
{
	setModel(_model);

	connect(_model, &PVRush::PVERFTreeModel::indexStateChanged, [=,this](QModelIndex index) {
		Qt::CheckState state =
		    static_cast<Qt::CheckState>(_model->data(index, Qt::CheckStateRole).toUInt());
		if (not _changing_check_state) {
			_changing_selection = true;
			set_children_state(index, state);
			_changing_selection = false;
			_changing_check_state = true;
			set_parents_state(index);
			_changing_check_state = false;

			Q_EMIT model_changed();
		}
	});

	connect(selectionModel(), &QItemSelectionModel::selectionChanged,
	        [=,this](const QItemSelection& sel, const QItemSelection& desel) {
		        if (not _changing_selection) {
			        auto change_states = [&](const QModelIndexList indexes, Qt::CheckState state) {
				        if (indexes.isEmpty()) {
					        return;
				        }
				        for (const QModelIndex& index : indexes) {
					        set_item_state(index, state);
				        }
				        _changing_check_state = true;
				        // Assume that all indexes have the same parent
				        set_parents_state(indexes.front());
				        _changing_check_state = false;
			        };
			        change_states(sel.indexes(), Qt::Checked);
			        change_states(desel.indexes(), Qt::Unchecked);

			        Q_EMIT model_changed();
		        }
	        });
}

void PVRush::PVERFTreeView::select(const rapidjson::Document& json)
{
	rapidjson::Document::AllocatorType alloc;

	_model->visit_nodes<const rapidjson::Value>(
	    PVERFTreeModel::ENodesType::ALL, &json,
	    [&](QModelIndex index, const rapidjson::Value* parent) {
		    PVERFTreeItem* item = static_cast<PVERFTreeItem*>(index.internalPointer());
		    const std::string& name = index.data().toString().toStdString();

		    if (parent->IsObject()) {
			    if (parent->HasMember(name.c_str())) {
				    const rapidjson::Value* child = &(*parent)[name.c_str()];
				    if (item->type() != PVERFTreeItem::EType::STATES) {
					    return child;
				    } else {
					    std::vector<std::pair<size_t, size_t>> l =
					        PVCore::deserialize_numbers_as_ranges(child->GetString());
					    QItemSelection selection;
					    for (auto [begin, end] : l) {
						    selection.select(_model->index(begin, 0, index),
						                     _model->index(end, 0, index));
					    }
					    selectionModel()->select(selection, QItemSelectionModel::Select);
				    }
			    }
		    } else if (parent->IsArray()) {
			    rapidjson::Value node_name;
			    node_name.SetString(name.c_str(), alloc);
			    auto array = parent->GetArray();
			    auto result = std::find(array.begin(), array.end(), node_name);
			    if (result != array.end()) {
				    selectionModel()->select(index, QItemSelectionModel::Select);
			    }
		    }

		    return (const rapidjson::Value*)nullptr;
	    });
}

void PVRush::PVERFTreeView::set_item_state(QModelIndex index, Qt::CheckState state)
{
	if (not index.isValid()) {
		return;
	}

	PVERFTreeItem* item = static_cast<PVERFTreeItem*>(index.internalPointer());
	if (item->is_node()) {
		_model->setData(index, QVariant(state), Qt::CheckStateRole);
	} else {
		item->set_state(state); // no need to triger a state changed signal (has performance impact)
	}
}

void PVRush::PVERFTreeView::set_children_state(QModelIndex index, Qt::CheckState state)
{
	if (not index.isValid()) {
		return;
	}
	QItemSelection selection;
	selection.select(_model->index(0, 0, index),
	                 _model->index(_model->rowCount(index) - 1, 0, index));
	selectionModel()->select(selection, state == Qt::Checked ? QItemSelectionModel::Select
	                                                         : QItemSelectionModel::Deselect);
	for (int i = 0; i < _model->rowCount(index); i++) {
		QModelIndex child = _model->index(i, 0, index);
		set_item_state(child, state);
	}
}

void PVRush::PVERFTreeView::set_parents_state(QModelIndex index)
{
	if (not index.isValid()) {
		return;
	}

	QModelIndex parent = index.parent();
	if (not parent.isValid()) {
		return;
	}

	PVERFTreeItem* item = static_cast<PVERFTreeItem*>(index.internalPointer());
	PVERFTreeItem* parent_item = static_cast<PVERFTreeItem*>(parent.internalPointer());
	int chid_count = _model->rowCount(parent);
	int selected_child_count = parent_item->selected_child_count();
	Qt::CheckState parent_state =
	    item->state() == Qt::PartiallyChecked
	        ? Qt::PartiallyChecked
	        : selected_child_count == 0
	              ? Qt::Unchecked
	              : (chid_count == selected_child_count ? Qt::Checked : Qt::PartiallyChecked);

	set_item_state(parent, parent_state);
	set_parents_state(parent);
}