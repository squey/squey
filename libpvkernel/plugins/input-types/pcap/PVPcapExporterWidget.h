/**
 * @file
 *
 * @copyright (C) ESI Group INENDI April 2015-2018
 */

#ifndef __PVKERNEL_PVPCAPEXPORTERWIDGET_H__
#define __PVKERNEL_PVPCAPEXPORTERWIDGET_H__

#include <pvkernel/widgets/PVExporterWidgetInterface.h>
#include "PVPcapExporter.h"

namespace PVGuiQt
{

class PVPcapExporterWidget : public PVWidgets::PVExporterWidgetInterface
{
  public:
	PVPcapExporterWidget(const PVRush::PVInputType::list_inputs& inputs, PVRush::PVNraw const& nraw)
	    : _exporter(PVRush::PVPcapExporter(inputs, nraw))

	{
		QVBoxLayout* layout = new QVBoxLayout;

		QCheckBox* complete_streams = new QCheckBox("Export complete TCP streams");
		complete_streams->setChecked(_exporter.get_export_complete_stream());
		QObject::connect(complete_streams, &QCheckBox::stateChanged,
		                 [&](int state) { _exporter.set_export_complete_stream((bool)state); });
		complete_streams->setChecked(_exporter.get_export_complete_stream());

		QCheckBox* open_pcap = new QCheckBox("Open PCAP after export");
		QObject::connect(open_pcap, &QCheckBox::stateChanged,
		                 [&](int state) { _exporter.set_open_pcap_after_export((bool)state); });
		open_pcap->setChecked(_exporter.get_open_pcap_after_export());

		layout->addWidget(complete_streams);
		layout->addWidget(open_pcap);
		layout->addSpacerItem(new QSpacerItem(1, 1, QSizePolicy::Minimum, QSizePolicy::Expanding));

		setLayout(layout);
	}

  public:
	PVRush::PVPcapExporter& exporter() override { return _exporter; }

  private:
	PVRush::PVPcapExporter _exporter;
};

} // namespace PVGuiQt

#endif // __PVKERNEL_PVPCAPEXPORTERWIDGET_H__
