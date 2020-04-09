/**
 * @file
 *
 *
 * @copyright (C) ESI Group INENDI 2017
 */

#ifndef INENDI_PVINPUTTYPEPCAP_H
#define INENDI_PVINPUTTYPEPCAP_H

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

	QIcon icon() const override { return QIcon(":/import-icon-white"); }
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
