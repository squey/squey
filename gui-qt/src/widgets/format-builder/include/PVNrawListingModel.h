/**
 * \file PVNrawListingModel.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVNRAWLISTINGMODEL_H
#define PVNRAWLISTINGMODEL_H

#include <pvkernel/core/general.h>

#include <QAbstractTableModel>
#include <QVariant>

// Forward declaration
namespace PVRush {
class PVNraw;
}

namespace PVInspector {

class PVNrawListingModel: public QAbstractTableModel
{
	Q_OBJECT

public:
	PVNrawListingModel(QObject* parent = NULL);

public:
	QVariant data(const QModelIndex &index, int role) const;
    QVariant headerData(int section, Qt::Orientation orientation, int role) const;
    int rowCount(const QModelIndex &index) const;
    int columnCount(const QModelIndex &index) const;
    Qt::ItemFlags flags(const QModelIndex &index) const;
	void sel_visible(bool visible);
	void set_selected_column(PVCol col);

public:
	void set_consistent(bool c);
	bool is_consistent();
	void set_nraw(PVRush::PVNraw const& nraw);

protected:
	const PVRush::PVNraw* _nraw;
	bool _is_consistent;
	PVCol _col_tosel;
	bool _show_sel;
};

}

#endif
