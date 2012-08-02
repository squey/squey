/**
 * \file PVViewsModel.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVVIEWSMODEL_H
#define PVVIEWSMODEL_H

#include <pvkernel/core/general.h>
#include <picviz/PVSource.h>
#include <picviz/PVPtrObjects.h>

#include <QAbstractItemModel>

namespace PVInspector {

class PVViewsModel: public QAbstractItemModel
{
public:
	class PVIndexNode
	{
	public:
		PVIndexNode(Picviz::PVMapped* mapped)
		{ _plotted = NULL; _mapped = mapped; }

		PVIndexNode(Picviz::PVPlotted* plotted)
		{ _plotted = plotted; _mapped = NULL; }

	public:
		bool is_mapped() const { return _mapped != NULL; }
		bool is_plotted() const { return _plotted != NULL; }

		Picviz::PVMapped* as_mapped() const { return _mapped; }
		Picviz::PVPlotted* as_plotted() const { return _plotted; }

		bool operator==(const PVIndexNode& other) const { return (_mapped == other._mapped) && (_plotted == other._plotted); }

	protected:
		Picviz::PVMapped* _mapped;
		Picviz::PVPlotted* _plotted;
	};
public:
	PVViewsModel(Picviz::PVSource const& src, QObject* parent = 0);
	~PVViewsModel();

public:
	QVariant data(const QModelIndex &index, int role) const;
    int rowCount(const QModelIndex &index) const;
    int columnCount(const QModelIndex &index) const;
	QModelIndex parent(const QModelIndex & index) const;
	QModelIndex index(int row, int column, const QModelIndex& parent = QModelIndex()) const;
	Qt::ItemFlags flags(const QModelIndex& index) const;

public:
	void force_refresh();
	void emitDataChanged(QModelIndex const& index);
	QModelIndex get_index_from_node(PVIndexNode const& node);

public:
	PVIndexNode const& get_object(QModelIndex const& index) const;

protected:
	Picviz::PVSource const& _src;
	Picviz::PVSource::list_mapped_t const& _mappeds;
	mutable QList<PVIndexNode*> _nodes_todel;
};

};

#endif

