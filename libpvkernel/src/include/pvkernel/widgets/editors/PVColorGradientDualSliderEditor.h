/**
 * \file PVColorGradientDualSliderEditor.h
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#ifndef PVCOLORGRADIENTDUALSLIDEREDITOR_H
#define PVCOLORGRADIENTDUALSLIDEREDITOR_H

#include <QWidget>
#include <pvkernel/core/general.h>
#include <pvkernel/core/PVColorGradientDualSliderType.h>
#include <pvkernel/widgets/PVColorPicker.h>


namespace PVWidgets {
class PVMainWindow;

/**
 * \class PVColorGradientDualSliderEditor
 */
class PVColorGradientDualSliderEditor : public PVColorPicker
{
	Q_OBJECT
	Q_PROPERTY(PVCore::PVColorGradientDualSliderType _color_slider READ get_values WRITE set_values USER true)

public:
	/**
	 * Constructor
	 */
	PVColorGradientDualSliderEditor(QWidget *parent = 0);

public:
	PVCore::PVColorGradientDualSliderType get_values() const;
	void set_values(PVCore::PVColorGradientDualSliderType v);
};

}

#endif // PVColorGradientDualSliderEditor_H
