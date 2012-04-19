#ifndef __PVAD2GRFFLISTMODEL_H
#define __PVAD2GRFFLISTMODEL_H

#include <QtCore/qstringlist.h>
#include <QtGui/qabstractitemview.h>

#include <picviz/PVTFViewRowFiltering.h>

namespace PVWidgets {

class PVAD2GRFFListModel : public QAbstractListModel
{
    Q_OBJECT
public:
    explicit PVAD2GRFFListModel(QObject *parent = 0);
    PVAD2GRFFListModel(const Picviz::PVView& src_view, const Picviz::PVView& dst_view, const Picviz::PVTFViewRowFiltering::list_rff_t &rffs, QObject *parent = 0);

    int rowCount(const QModelIndex &parent = QModelIndex()) const;

    QVariant data(const QModelIndex &index, int role) const;
    bool setData(const QModelIndex &index, const QVariant &value, int role = Qt::EditRole);

    Qt::ItemFlags flags(const QModelIndex &index) const;

    void addRow(QModelIndex model_index, Picviz::PVSelRowFilteringFunction_p rff);
    bool insertRows(int row, int count, const QModelIndex &parent = QModelIndex());
    bool removeRows(int row, int count, const QModelIndex &parent = QModelIndex());

    //void sort(int column, Qt::SortOrder order = Qt::AscendingOrder);

    Picviz::PVTFViewRowFiltering::list_rff_t getRFFList() const;
    void setRFFList(const Picviz::PVTFViewRowFiltering::list_rff_t &rffs);

    Qt::DropActions supportedDropActions() const;

private:
    Q_DISABLE_COPY(PVAD2GRFFListModel)
    Picviz::PVTFViewRowFiltering::list_rff_t _rffs;
    const Picviz::PVView* _src_view;
    const Picviz::PVView* _dst_view;
};

}


#endif // __PVAD2GRFFLISTMODEL_H
