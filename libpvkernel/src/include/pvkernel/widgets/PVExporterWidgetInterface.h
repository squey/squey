/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2018
 */

#ifndef __PVGUIQT_PVEXPORTERWIDGETINTERFACE_H__
#define __PVGUIQT_PVEXPORTERWIDGETINTERFACE_H__

#include <pvkernel/rush/PVExporter.h>

#include <QWidget>

namespace PVWidgets
{

class PVExporterWidgetInterface : public QWidget
{
  public:
	PVExporterWidgetInterface() {}
	virtual ~PVExporterWidgetInterface(){};

  public:
	virtual PVRush::PVExporterBase& exporter() = 0;
};

} // namespace PVWidgets

#endif // __PVGUIQT_PVEXPORTERWIDGETINTERFACE_H__
