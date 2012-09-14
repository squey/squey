/**
 * \file PVLayerStackWidget.h
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#ifndef PVLAYERSTACKWIDGET_H
#define PVLAYERSTACKWIDGET_H

#include <QToolBar>
#include <QWidget>

#include <picviz/PVView_types.h>

namespace PVGuiQt {

class PVLayerStackModel;
class PVLayerStackView;

/**
 *  \class PVLayerStackWidget
 */
class PVLayerStackWidget : public QWidget
{
	Q_OBJECT

public:
	PVLayerStackWidget(Picviz::PVView_sp& lib_view, QWidget* parent = NULL);

public:
	PVLayerStackView *get_layer_stack_view() const { return _layer_stack_view; }

private:
	void create_actions(QToolBar* toolbar);
	PVLayerStackModel* ls_model();

private slots:
	void delete_layer();
	void duplicate_layer();
	void move_down();
	void move_up();
	void new_layer();

private:
	PVLayerStackView *_layer_stack_view;
};


}

#endif // PVLAYERSTACKWIDGET_H


