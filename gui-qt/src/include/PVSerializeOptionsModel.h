/**
 * \file PVSerializeOptionsModel.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVSERIALIZEOPTIONSMODEL_H
#define PVSERIALIZEOPTIONSMODEL_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVSerializeArchiveOptions_types.h>
#include <pvkernel/core/PVSerializeObject.h>

#include <QAbstractItemModel>

namespace PVInspector {

class PVSerializeOptionsModel: public QAbstractItemModel
{
public:
	PVSerializeOptionsModel(PVCore::PVSerializeArchiveOptions_p options, QObject* parent = 0);

public:
	QVariant data(const QModelIndex &index, int role) const;
    //QVariant headerData(int section, Qt::Orientation orientation, int role) const;
    int rowCount(const QModelIndex &index) const;
    int columnCount(const QModelIndex &index) const;
	QModelIndex parent(const QModelIndex & index) const;
	QModelIndex index(int row, int column, const QModelIndex& parent = QModelIndex()) const;
	Qt::ItemFlags flags(const QModelIndex& index) const;
	bool setData(const QModelIndex& index, const QVariant& value, int role = Qt::EditRole);

protected:
	PVCore::PVSerializeObject::list_childs_t const& get_childs_index(const QModelIndex& parent) const;
	PVCore::PVSerializeObject* get_so_index(const QModelIndex& index) const;
	void emitDataChangedChildren(const QModelIndex& index);

protected:
	PVCore::PVSerializeArchiveOptions_p _options;
};

};

#endif

