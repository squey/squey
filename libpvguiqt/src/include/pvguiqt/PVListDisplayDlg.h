/**
 * \file PVListDisplayDlg.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVGUIQT_PVLISTDISPLAYDLG_H
#define PVGUIQT_PVLISTDISPLAYDLG_H

#include <picviz/PVView_types.h>
#include <pvguiqt/ui/PVListDisplayDlg.h>

#include <QAbstractListModel>
#include <QVector>
#include <QDialog>
#include <QFileDialog>

namespace PVGuiQt {

class PVStringSortProxyModel;

class PVListDisplayDlg: public QDialog, Ui::PVListDisplayDlg
{
	Q_OBJECT

public:
	PVListDisplayDlg(QAbstractListModel* model,  QWidget* parent = NULL);

public:
	void set_description(QString const& desc);

protected:
	QAbstractListModel* model();
	PVStringSortProxyModel* proxy_model();

private slots:
	void copy_to_clipboard();
	void copy_value_clipboard();
	void copy_to_file() { write_to_file_ui(false); }
	void append_to_file() { write_to_file_ui(true); }
	void sort();

private:
	void write_to_file_ui(bool append);
	void write_to_file(QFile& file);
	bool write_values(QDataStream* stream);

private:
	QFileDialog _file_dlg;
};

}

#endif
