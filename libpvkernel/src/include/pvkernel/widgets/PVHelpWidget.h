/**
 * @file
 *
 * @copyright (C) Picviz Labs 2013-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVWIDGETS_PVHELPWIDGET_H
#define PVWIDGETS_PVHELPWIDGET_H

#include <pvkernel/widgets/PVTextPopupWidget.h>

namespace PVWidgets
{

class PVHelpWidget : public PVTextPopupWidget
{
  public:
	explicit PVHelpWidget(QWidget* parent);

	/**
	 * test if key is one of the close keys
	 *
	 * @param key the key to test
	 *
	 * @return true if key is one of the help key; false otherwise
	 *
	 * @see PVWidgets::PVHelpWidget*
	 */
	bool is_close_key(int key) override;

  public:
	/**
	 * test if key is one of the help keys
	 *
	 * @param key the key to test
	 *
	 * @return true if key is one of the help key; false otherwise
	 */
	static bool is_help_key(int key);
};
} // namespace PVWidgets

#endif // PVWIDGETS_PVHELPWIDGET_H
