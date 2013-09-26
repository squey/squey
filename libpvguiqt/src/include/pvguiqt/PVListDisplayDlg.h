/**
 * \file PVListDisplayDlg.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVGUIQT_PVLISTDISPLAYDLG_H
#define PVGUIQT_PVLISTDISPLAYDLG_H

#include <pvguiqt/ui/PVListDisplayDlg.h>

#include <pvhive/PVActor.h>
#include <pvhive/PVObserverSignal.h>

#include <picviz/PVView_types.h>

#include <pvkernel/core/PVArgument.h>

#include <QAbstractListModel>
#include <QVector>
#include <QDialog>
#include <QFileDialog>

namespace PVGuiQt {

class PVLayerFilterProcessWidget;

class PVStringSortProxyModel;

class PVListDisplayDlg: public QDialog, public Ui::PVListDisplayDlg
{
	Q_OBJECT

public:
	PVListDisplayDlg(QAbstractListModel* model,  QWidget* parent = NULL);

public:
	void set_description(QString const& desc);

protected:
	QAbstractListModel* model();
	PVStringSortProxyModel* proxy_model();

protected:
	virtual void process_context_menu(QAction* act);
	virtual void process_hhead_context_menu(QAction* act);

protected slots:
	void section_pressed(int col);
	void section_clicked(int col);

protected:
	virtual void sort_by_column(int col);

private slots:
	void copy_to_clipboard();
	void copy_value_clipboard();
	void copy_to_file() { write_to_file_ui(false); }
	void append_to_file() { write_to_file_ui(true); }
	void sort();
	void show_ctxt_menu(const QPoint& pos);
	void show_hhead_ctxt_menu(const QPoint& pos);

private:
	void write_to_file_ui(bool append);
	void write_to_file(QFile& file);
	bool write_values(QDataStream* stream);

protected:
	QFileDialog _file_dlg;
	QAction* _copy_values_act;
	QMenu* _ctxt_menu;
	QMenu* _hhead_ctxt_menu;
	PVGuiQt::PVLayerFilterProcessWidget* _ctxt_process = nullptr;
	PVCore::PVArgumentList _ctxt_args;
	//QItemSelection _item_selection;
};

}

#endif
