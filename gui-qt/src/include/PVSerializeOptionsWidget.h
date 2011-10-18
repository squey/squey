#ifndef PVSERIALIZEOPTIONSWIDGET_H
#define PVSERIALIZEOPTIONSWIDGET_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVSerializeArchiveOptions_types.h>

#include "PVSerializeOptionsModel.h"

#include <QWidget>
#include <QTreeView>

namespace PVInspector {

class PVSerializeOptionsWidget: public QWidget
{
public:
	PVSerializeOptionsWidget(PVCore::PVSerializeArchiveOptions_p options, QWidget* parent = 0);

protected:
	QTreeView* _view;
	PVSerializeOptionsModel* _model;
	
	Q_OBJECT
};

}

#endif
