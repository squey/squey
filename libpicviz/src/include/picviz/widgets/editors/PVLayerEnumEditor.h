/**
 * \file PVLayerEnumEditor.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVCORE_PVLAYERENUMEDITOR_H
#define PVCORE_PVLAYERENUMEDITOR_H

#include <QComboBox>

#include <pvkernel/core/general.h>
#include <picviz/PVView.h>

namespace PVWidgets {

class PVLayerEnumEditor : public QComboBox
{
	Q_OBJECT
	Q_PROPERTY(Picviz::PVLayer* _layer READ get_layer WRITE set_layer USER true)

public:
	PVLayerEnumEditor(Picviz::PVView const& view, QWidget *parent = 0);
	virtual ~PVLayerEnumEditor();

public:
	Picviz::PVLayer* get_layer() const;
	void set_layer(Picviz::PVLayer* l);

protected:
	Picviz::PVView const& _view;
};

}

#endif
