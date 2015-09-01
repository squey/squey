/**
 * \file PVAD2GRFFListModel.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef __PVAD2GRFFLISTMODEL_H
#define __PVAD2GRFFLISTMODEL_H

#include <QtCore/qstringlist.h>
#include <QtWidgets/qabstractitemview.h>
#include <QtCore/QAbstractTableModel>

#include <pvkernel/core/PVBinaryOperation.h>
#include <picviz/PVTFViewRowFiltering.h>

namespace PVWidgets {

class PVAD2GRFFListModel : public QAbstractTableModel
{
    Q_OBJECT
public:
    PVAD2GRFFListModel(const Picviz::PVView& src_view, const Picviz::PVView& dst_view, Picviz::PVTFViewRowFiltering::list_rff_t &rffs, QObject *parent = 0);

    int rowCount(const QModelIndex &parent = QModelIndex()) const;
    int columnCount(const QModelIndex &parent = QModelIndex()) const;

    QVariant data(const QModelIndex &index, int role) const;
    bool setData(const QModelIndex &index, const QVariant &value, int role = Qt::EditRole);

    Qt::ItemFlags flags(const QModelIndex &index) const;

    void addRow(QModelIndex model_index, Picviz::PVSelRowFilteringFunction_p rff);
    bool insertRows(int row, int count, const QModelIndex &parent = QModelIndex());
    bool removeRows(int row, int count, const QModelIndex &parent = QModelIndex());

    void move_index(QModelIndex index, bool up);

    //void sort(int column, Qt::SortOrder order = Qt::AscendingOrder);

    Picviz::PVTFViewRowFiltering::list_rff_t& get_rffs() const;

    //Qt::DropActions supportedDropActions() const;

private:
    Picviz::PVTFViewRowFiltering::list_rff_t& _rffs;
    const Picviz::PVView& _src_view;
    const Picviz::PVView& _dst_view;
};

}


#endif // __PVAD2GRFFLISTMODEL_H
