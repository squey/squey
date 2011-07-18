//! \file PVArgumentListModel.h
//! $Id: PVArgumentListModel.h 3090 2011-06-09 04:59:46Z stricaud $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVARGUMENTLISTMODEL_H
#define PVARGUMENTLISTMODEL_H

#include <QtCore>

#include <QAbstractTableModel>
#include <QVariant>

#include <pvfilter/PVArgument.h>
#include <picviz/general.h>

namespace PVInspector {

class PVArgumentListModel : public QAbstractTableModel {
public:
	PVArgumentListModel(PVFilter::PVArgumentList& args, QObject* parent = 0);
public:
	int rowCount(const QModelIndex &parent) const;
	int columnCount(const QModelIndex &parent) const;
	QVariant data(const QModelIndex& index, int role) const;
	bool setData(const QModelIndex& index, const QVariant &value, int role);
	Qt::ItemFlags flags(const QModelIndex& index) const;
	QVariant headerData(int section, Qt::Orientation orientation, int role) const; 
protected:
	QString                    _header_name[2];
	PVFilter::PVArgumentList&  _args;
};


}

#endif	/* PVARGUMENTLISTMODEL_H */

