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

#ifndef SQUEY_PVINPUTTYPEPCAP_H
#define SQUEY_PVINPUTTYPEPCAP_H

#include <pvkernel/rush/PVInputType.h>

#include "PVPcapParamsWidget.h"
#include "PVPcapExporter.h"
#include "PVPcapExporterWidget.h"
#include "pcap/PVPcapDescription.h"

#include <QString>
#include <QStringList>
#include <QIcon>
#include <QCursor>
#include <QSet>

namespace PVPcapsicum
{

class PVInputTypePcap : public PVRush::PVInputTypeDesc<PVRush::PVPcapDescription>
{
  public:
	PVInputTypePcap();
	virtual ~PVInputTypePcap();

  public:
	bool createWidget(PVRush::hash_formats& formats,
	                  PVRush::PVInputType::list_inputs& inputs,
	                  QString& format,
	                  PVCore::PVArgumentList& args_ext,
	                  QWidget* parent = nullptr) const override;

	/* exporter */
	std::unique_ptr<PVRush::PVExporterBase>
	create_exporter(const list_inputs& inputs, PVRush::PVNraw const& nraw) const override;
	PVWidgets::PVExporterWidgetInterface*
	create_exporter_widget(const list_inputs& inputs, PVRush::PVNraw const& nraw) const override;
	QString get_exporter_filter_string(const list_inputs& inputs) const override;

	QString name() const override;
	QString human_name() const override;
	QString human_name_serialize() const override;
	QString internal_name() const override;
	QString menu_input_name() const override;
	QString tab_name_of_inputs(PVRush::PVInputType::list_inputs const& in) const override;
	QKeySequence menu_shortcut() const override;
	bool get_custom_formats(PVRush::PVInputDescription_p in,
	                        PVRush::hash_formats& formats) const override;

	QIcon icon() const override { return QIcon(":/pcap_icon"); }
	QCursor cursor() const override { return QCursor(Qt::PointingHandCursor); }

  public:
	QStringList input_paths() const { return _input_paths; }

  protected:
	bool load_files(pvpcap::splitted_files_t&& filenames,
	                PVRush::PVInputType::list_inputs& inputs,
	                QWidget* parent) const;

  protected:
	mutable QSet<QString> _tmp_dir_to_delete;
	int _limit_nfds;
	mutable QStringList _input_paths;

  protected:
	CLASS_REGISTRABLE_NOCOPY(PVPcapsicum::PVInputTypePcap)
};
} // namespace PVPcapsicum

#endif
