//! \file PVArgumentListWidget.h
//! $Id: PVArgumentListWidget.h 3199 2011-06-24 07:14:57Z aguinet $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVARGUMENTLISTWIDGET_H
#define PVARGUMENTLISTWIDGET_H

#include <QtCore>
#include <QDialog>
#include <QEvent>


#include <QTableView>
#include <QListWidget>
#include <QVariant>
#include <QHBoxLayout>

#include <pvkernel/core/PVArgument.h>
#include <picviz/general.h>
#include <picviz/PVView.h>

#include <PVArgumentListModel.h>
#include <PVArgumentListDelegate.h>

namespace PVInspector {
class PVMainWindow;

class PVArgumentListWidget: public QDialog
{
	Q_OBJECT

public:
	PVArgumentListWidget(Picviz::PVView& view, PVCore::PVArgumentList &args, QWidget* parent);
	virtual ~PVArgumentListWidget();
	bool eventFilter(QObject *obj, QEvent *event);
	void init();
	inline bool args_changed() { return _args_has_changed; }
	inline void clear_args_state() { _args_has_changed = false; }

private slots:
	void args_changed_Slot();

protected:
	virtual void create_btns();
	virtual void set_btns_layout();
	virtual void connect_btns();


/* public slots: */
/* 	void widget_clicked_Slot(); */

protected:
	QTableView*               _args_view;
	PVArgumentListModel*      _args_model;
	PVArgumentListDelegate*   _args_del;
	PVCore::PVArgumentList& _args;
	Picviz::PVView&           _view;

	// Standard buttons
	QPushButton*              _apply_btn;
	QPushButton*              _cancel_btn;
	QHBoxLayout*              _btn_layout;

private:
	bool _args_has_changed;
};


}

#endif



