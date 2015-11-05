/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVCOLORDIALOG_H
#define PVCOLORDIALOG_H

#include <QColorDialog>
#include <inendi/PVView_types.h>

namespace PVInspector {
class PVMainWindow;

/**
 * \class PVColorDialog
 */
class PVColorDialog : public QColorDialog
{
	Q_OBJECT

public:
	/**
	 * Constructor
	 */
	PVColorDialog(Inendi::PVView& inendi_view, QWidget* parent = 0);

public:
	Inendi::PVView& get_lib_view() { return _inendi_view; }

protected:
	Inendi::PVView& _inendi_view;
};
}

#endif // PVCOLORDIALOG_H

