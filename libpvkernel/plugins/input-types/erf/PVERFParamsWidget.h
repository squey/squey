/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2019
 */

#ifndef __PVERFPARAMSWIDGET_H__
#define __PVERFPARAMSWIDGET_H__

#include <QDialog>

namespace PVRush
{

class PVInputTypeERF;

class PVERFParamsWidget : public QDialog
{
	Q_OBJECT

  public:
	PVERFParamsWidget(PVInputTypeERF const* in_t, QWidget* parent);
};

} // namespace PVRush

#endif // __PVERFPARAMSWIDGET_H__
