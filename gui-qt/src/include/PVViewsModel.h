/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVVIEWSMODEL_H
#define PVVIEWSMODEL_H

#include <pvkernel/core/general.h>
#include <inendi/PVSource.h>
#include <inendi/PVPtrObjects.h>

#include <QAbstractItemModel>

namespace PVInspector {

class PVViewsModel: public QAbstractItemModel
{
public:
	class PVIndexNode
	{
	public:
		PVIndexNode(Inendi::PVMapped* mapped)
		{ _plotted = NULL; _mapped = mapped; }

		PVIndexNode(Inendi::PVPlotted* plotted)
		{ _plotted = plotted; _mapped = NULL; }

	public:
		bool is_mapped() const { return _mapped != NULL; }
		bool is_plotted() const { return _plotted != NULL; }

		Inendi::PVMapped* as_mapped() const { return _mapped; }
		Inendi::PVPlotted* as_plotted() const { return _plotted; }

		bool operator==(const PVIndexNode& other) const { return (_mapped == other._mapped) && (_plotted == other._plotted); }

	protected:
		Inendi::PVMapped* _mapped;
		Inendi::PVPlotted* _plotted;
	};
public:
	PVViewsModel(Inendi::PVSource const& src, QObject* parent = 0);
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
	Inendi::PVSource const& _src;
	mutable QList<PVIndexNode*> _nodes_todel;
};

};

#endif

