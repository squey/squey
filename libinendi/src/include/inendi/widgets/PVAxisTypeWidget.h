/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVWIDGETS_PVAXISTYPEWIDGET_H
#define PVWIDGETS_PVAXISTYPEWIDGET_H

#include <pvkernel/widgets/PVComboBox.h>

namespace PVWidgets
{

/**
 * This widget is a combo box showing types names.
 */
class PVAxisTypeWidget : public PVComboBox
{
  public:
	/**
	 * Build the combo box with only type that have same storage as current_type.
	 *
	 * It current_type is "all", then all types can be use.
	 */
	PVAxisTypeWidget(QString const& current_type, QWidget* parent = nullptr);

  public:
	inline QString get_sel_type() const { return currentText(); }
	inline bool sel_type(QString const& type) { return select(type); }
};
}

#endif
