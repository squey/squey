#include "PVViewsModel.h"
#include <picviz/PVMapped.h>
#include <picviz/PVPlotted.h>
#include <picviz/PVSource.h>

#include <QFont>

PVInspector::PVViewsModel::PVViewsModel(Picviz::PVSource const& src, QObject* parent):
	QAbstractItemModel(parent),
	_src(src),
	_mappeds(src.get_mappeds())
{
}

PVInspector::PVViewsModel::~PVViewsModel()
{
	QList<PVIndexNode*>::iterator it;
	for (it = _nodes_todel.begin(); it != _nodes_todel.end(); it++) {
		delete *it;
	}
}

PVInspector::PVViewsModel::PVIndexNode const& PVInspector::PVViewsModel::get_object(QModelIndex const& index) const
{
	assert(index.isValid());
	return *(static_cast<PVIndexNode*>(index.internalPointer()));
}

QModelIndex PVInspector::PVViewsModel::index(int row, int column, const QModelIndex& parent) const
{
	// Column is always 0 (see columnCount), but asserts it
	assert(column == 0);

	if (!parent.isValid()) {
		// Root elements are mapped objects
		assert(row < _mappeds.size());
		Picviz::PVMapped_p mapped(_mappeds.at(row));
		PVIndexNode* nt = new PVIndexNode(mapped.get());
		_nodes_todel.push_back(nt);
		return createIndex(row, column, nt);
	}

	PVIndexNode node_obj = get_object(parent);
	assert(node_obj.is_mapped());
	// Create an index for a plotted child object
	Picviz::PVMapped::list_plotted_t const& children = node_obj.as_mapped()->get_plotteds();
	assert(row < children.size());
	PVIndexNode* nt = new PVIndexNode(children.at(row).get());
	_nodes_todel.push_back(nt);
	return createIndex(row, column, nt);
}

int PVInspector::PVViewsModel::rowCount(const QModelIndex &index) const
{
	if (!index.isValid()) {
		// Return number of mapped
		return _mappeds.size();
	}
	
	PVIndexNode node_obj = get_object(index);
	if (node_obj.is_plotted()) {
		// That's a plotted object, so no children
		return 0;
	}

	assert(index.row() < _mappeds.size());
	// Return the number of plotted children objects
	return _mappeds.at(index.row())->get_plotteds().size();
}

int PVInspector::PVViewsModel::columnCount(const QModelIndex& /*index*/) const
{
	// We have only one column
	return 1;
}

QVariant PVInspector::PVViewsModel::data(const QModelIndex &index, int role) const
{
	if (!index.isValid()) {
		return QVariant();
	}
	PVIndexNode node_obj = get_object(index);
	switch (role) {
		case Qt::DisplayRole:
		{
			QString ret;
			if (node_obj.is_mapped()) {
				Picviz::PVMapped* mapped = node_obj.as_mapped();
				ret = QString("Mapped");
				ret += ": " + mapped->get_name();
				if (!mapped->is_uptodate()) {
					ret += QString(" *");
				}
			}
			else {
				Picviz::PVPlotted* plotted = node_obj.as_plotted();
				ret = QString("Plotted");
				ret += ": " + plotted->get_name();
				if (!node_obj.as_plotted()->is_uptodate()) {
					ret += QString(" *");
				}
			}
			return ret;
		}
		case Qt::FontRole:
		{
			if (!node_obj.is_plotted()) {
				return QVariant();
			}
			Picviz::PVPlotted* plotted = node_obj.as_plotted();
			if (plotted->get_view() == _src.current_view()) {
				QFont font;
				font.setBold(true);
				return font;
			}
			return QVariant();
		}
		default:
			break;
	};

	return QVariant();
}

Qt::ItemFlags PVInspector::PVViewsModel::flags(const QModelIndex& /*index*/) const
{
	Qt::ItemFlags flags = Qt::ItemIsSelectable | Qt::ItemIsEnabled | Qt::ItemIsUserCheckable;
	return flags;
}

QModelIndex PVInspector::PVViewsModel::parent(const QModelIndex & index) const
{
	if (!index.isValid()) {
		return QModelIndex();
	}

	PVIndexNode node_obj = get_object(index);
	if (node_obj.is_mapped()) {
		return QModelIndex();
	}

	Picviz::PVPlotted* plotted = node_obj.as_plotted();
	Picviz::PVMapped* mapped = plotted->get_mapped_parent();

	Picviz::PVSource::list_mapped_t::const_iterator it;
	bool found = false;
	int idx = 0;
	for (it = _mappeds.begin(); it != _mappeds.end(); it++) {
		if (it->get() == mapped) {
			found = true;
			break;
		}
		idx++;
	}
	assert(found);

	PVIndexNode* nt = new PVIndexNode(mapped);
	_nodes_todel.push_back(nt);
	return createIndex(idx, 0, nt);
}

void PVInspector::PVViewsModel::force_refresh()
{
	emit layoutChanged();
}

QModelIndex PVInspector::PVViewsModel::get_index_from_node(PVIndexNode const& node)
{
	int nmappeds = rowCount(QModelIndex());
	for (int i = 0; i < nmappeds; i++) {
		QModelIndex idx = index(i, 0, QModelIndex());
		if (node.is_mapped()) {
			PVIndexNode const& test = get_object(idx);
			if (test == node) {
				return idx;
			}
		}
		else {
			int nplotted = rowCount(idx);
			for (int j = 0; j < nplotted; j++) {
				QModelIndex idx_plotted = index(j, 0, idx);
				if (get_object(idx_plotted) == node) {
					return idx_plotted;
				}
			}
		}
	}
	return QModelIndex();
}
