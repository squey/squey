/* * MIT License
 *
 * © ESI Group, 2015
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
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
