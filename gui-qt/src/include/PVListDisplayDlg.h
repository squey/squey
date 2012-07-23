/**
 * \file PVListDisplayDlg.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVINSPECTOR_PVLISTDISPLAYDLG_H
#define PVINSPECTOR_PVLISTDISPLAYDLG_H

#include <picviz/PVView_types.h>
#include "../ui_PVListDisplayDlg.h"

#include <QAbstractListModel>
#include <QVector>
#include <QDialog>

namespace PVInspector {

class PVListDisplayDlg: public QDialog, Ui::PVListDisplayDlg
{
	Q_OBJECT

public:
	PVListDisplayDlg(QAbstractListModel* model,  QWidget* parent = NULL);

public:
	void set_description(QString const& desc);

private slots:
	void copy_to_clipboard();
	void copy_to_file();
	void copy_value_clipboard();

private:
	bool write_values(QDataStream* stream);

private:
	QAbstractListModel* _model;
};

}

#endif
