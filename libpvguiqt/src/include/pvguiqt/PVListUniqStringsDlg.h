/**
 * \file PVListColNrawDlg.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVGUIQT_PVLISTCOLNRAWDLG_H
#define PVGUIQT_PVLISTCOLNRAWDLG_H

#include <pvkernel/rush/PVNraw.h>

#include <picviz/PVView_types.h>

#include <pvguiqt/PVListDisplayDlg.h>

#include <QAbstractListModel>
#include <QDialog>

namespace PVGuiQt {

namespace __impl {

class PVListUniqStringsModel: public QAbstractListModel
{
	Q_OBJECT

public:
	PVListUniqStringsModel(PVRush::PVNraw::unique_values_t& values, QWidget* parent = NULL);

public:
	int rowCount(QModelIndex const& parent = QModelIndex()) const;
	QVariant data(QModelIndex const& index, int role = Qt::DisplayRole) const;
	QVariant headerData(int section, Qt::Orientation orientation, int role) const;

private:
	std::vector<std::string_tbb> _values;
};

}

class PVListUniqStringsDlg: public PVListDisplayDlg
{
	Q_OBJECT

public:
	PVListUniqStringsDlg(PVRush::PVNraw::unique_values_t& values, QWidget* parent = NULL);
	virtual ~PVListUniqStringsDlg();
};

}

#endif
