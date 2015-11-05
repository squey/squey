/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVCORE_PVLAYERENUMEDITOR_H
#define PVCORE_PVLAYERENUMEDITOR_H

#include <QComboBox>

#include <pvkernel/core/general.h>
#include <inendi/PVView.h>

namespace PVWidgets {

class PVLayerEnumEditor : public QComboBox
{
	Q_OBJECT
	Q_PROPERTY(Inendi::PVLayer* _layer READ get_layer WRITE set_layer USER true)

public:
	PVLayerEnumEditor(Inendi::PVView const& view, QWidget *parent = 0);
	virtual ~PVLayerEnumEditor();

public:
	Inendi::PVLayer* get_layer() const;
	void set_layer(Inendi::PVLayer* l);

protected:
	Inendi::PVView const& _view;
};

}

#endif
