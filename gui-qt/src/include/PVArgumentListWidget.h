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

#include <QDataWidgetMapper>
#include <QItemEditorFactory>
#include <QTableView>
#include <QListWidget>
#include <QVariant>
#include <QHBoxLayout>

#include <pvkernel/core/PVArgument.h>
#include <picviz/PVView_types.h>

#include <PVArgumentListModel.h>

namespace PVInspector {
class PVMainWindow;

class PVArgumentListWidget: public QWidget
{
	Q_OBJECT

public:
	PVArgumentListWidget(QWidget* parent = NULL);
	PVArgumentListWidget(QItemEditorFactory* args_widget_factory, QWidget* parent = NULL);
	PVArgumentListWidget(QItemEditorFactory* args_widget_factory, PVCore::PVArgumentList &args, QWidget* parent = NULL);
	virtual ~PVArgumentListWidget();
	//bool eventFilter(QObject *obj, QEvent *event);
	void set_args(PVCore::PVArgumentList& args);
	void set_widget_factory(QItemEditorFactory* factory);
	inline bool args_changed() { return _args_has_changed; }
	inline void clear_args_state() { _args_has_changed = false; }
	PVCore::PVArgumentList* get_args() { return _args; }


public:
	static QItemEditorFactory* create_layer_widget_factory(Picviz::PVView& view);
	static QItemEditorFactory* create_mapping_plotting_widget_factory();

private:
	void init_widgets();

private slots:
	void args_changed_Slot();

signals:
	void args_changed_Signal();

/* public slots: */
/* 	void widget_clicked_Slot(); */

protected:
	QGridLayout*              _args_layout;
	QDataWidgetMapper*        _mapper;
	QItemEditorFactory*       _args_widget_factory;
	PVArgumentListModel*      _args_model;
	PVCore::PVArgumentList*   _args;

	// Standard buttons
	QPushButton*              _apply_btn;
	QPushButton*              _cancel_btn;
	QHBoxLayout*              _btn_layout;

private:
	bool _args_has_changed;
};


}

#endif
