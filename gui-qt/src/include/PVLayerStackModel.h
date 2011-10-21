//! \file PVLayerStackModel.h
//! $Id: PVLayerStackModel.h 2573 2011-05-05 08:12:16Z aguinet $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVLAYERSTACKMODEL_H
#define PVLAYERSTACKMODEL_H

#include <QAbstractTableModel>

#include <QtGui>
#include <QtCore>

//#include <picviz/layer-stack.h>
#include <picviz/PVLayer.h>
#include <picviz/PVLayerStack.h>

namespace PVInspector {
class PVMainWindow;
class PVTabSplitter;

/**
 * \class PVLayerStackModel
 */
class PVLayerStackModel : public QAbstractTableModel
{
Q_OBJECT

	PVMainWindow         *main_window;     //!<
	PVTabSplitter        *parent_widget;   //!<

	Picviz::PVLayerStack &lib_layer_stack; //!<

	QBrush select_brush;       //!<
	QFont select_font;         //!<
	QBrush unselect_brush;     //!<
	QFont unselect_font;       //!<

public:
	/**
	*  Contructor.
	*
	*  @param mw
	*  @param parent
	*/
	PVLayerStackModel(PVMainWindow *mw, PVTabSplitter *parent);

	/**
	*
	* @return
	*/
	Picviz::PVLayerStack &get_layer_stack_lib()const{return lib_layer_stack;}

	/**
	*  @param
	*
	*  @return
	*/
	int columnCount(const QModelIndex &index) const;

	/**
	* @param index
	* @param role
	*
	* @return
	*/
	QVariant data(const QModelIndex &index, int role) const;

	/**
	*
	* @param index
	*
	* @return
	*/
	Qt::ItemFlags flags(const QModelIndex &index) const;

	/**
	*
	* @param section
	* @param orientation
	* @param role
	*
	* @return
	*/
	QVariant headerData(int section, Qt::Orientation orientation, int role) const;

	/**
	*
	* @param index
	*
	* @return
	*/
	int rowCount(const QModelIndex &index) const;

	/**
	*
	* @param index
	* @param value
	* @param role
	*
	* @return
	*/
	bool setData(const QModelIndex &index, const QVariant &value, int role);

	/**
	*
	*/
	void emit_layoutChanged();

	void update_layer_stack();
};
}

#endif
