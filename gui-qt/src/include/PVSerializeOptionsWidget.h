/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVSERIALIZEOPTIONSWIDGET_H
#define PVSERIALIZEOPTIONSWIDGET_H

#include "PVSerializeOptionsModel.h"

#include <QWidget>
#include <QTreeView>

namespace PVInspector
{

class PVSerializeOptionsWidget : public QWidget
{
  public:
	PVSerializeOptionsWidget(std::shared_ptr<PVCore::PVSerializeArchiveOptions> options,
	                         QWidget* parent = 0);

  public:
	QTreeView* get_view() { return _view; }

  protected:
	QTreeView* _view;
	PVSerializeOptionsModel* _model;
};
}

#endif
