/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVCOLORGRADIENTDUALSLIDEREDITOR_H
#define PVCOLORGRADIENTDUALSLIDEREDITOR_H

#include <pvkernel/core/PVColorGradientDualSliderType.h>
#include <pvkernel/widgets/PVColorPicker.h>

class QWidget;

namespace PVWidgets
{
class PVMainWindow;

/**
 * \class PVColorGradientDualSliderEditor
 */
class PVColorGradientDualSliderEditor : public PVColorPicker
{
	Q_OBJECT
	Q_PROPERTY(PVCore::PVColorGradientDualSliderType _color_slider READ get_values WRITE set_values
	               USER true)

  public:
	/**
	 * Constructor
	 */
	explicit PVColorGradientDualSliderEditor(QWidget* parent = 0);

  public:
	PVCore::PVColorGradientDualSliderType get_values() const;
	void set_values(PVCore::PVColorGradientDualSliderType v);
};
} // namespace PVWidgets

#endif // PVColorGradientDualSliderEditor_H
