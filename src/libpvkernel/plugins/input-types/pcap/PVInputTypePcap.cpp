//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include "PVInputTypePcap.h"

#include <pvbase/general.h>

#include <pvkernel/core/PVDirectory.h>
#include <pvkernel/core/PVUtils.h>
#include <pvkernel/rush/PVNrawCacheManager.h>

#include "pcap/PVPcapDescription.h"
#include "pcap/PVPcapExporterWidget.h"
#include "pcap/pcap-gui/ImportDialog.h"

#include <pvkernel/widgets/PVDialogImportWidget.h>

#include <QPushButton>
#include <QFileInfo>
#include <QTextStream>
#include <QCheckBox>

#include <stdexcept>

#include <libpvpcap/ws.h>

PVPcapsicum::PVInputTypePcap::PVInputTypePcap()
    : PVRush::PVInputTypeDesc<PVRush::PVPcapDescription>()
{
}

PVPcapsicum::PVInputTypePcap::~PVInputTypePcap()
{
	for (const QString& tmp_dir : _tmp_dir_to_delete) {
		PVLOG_INFO("Delete temporary directory %s...\n", qPrintable(tmp_dir));
		PVCore::PVDirectory::remove_rec(tmp_dir);
	}
}

PVWidgets::PVImportWidget* PVPcapsicum::PVInputTypePcap::create_import_widget(QWidget* parent) const
{
	auto* dialog = new ImportDialog({}, nullptr);
	auto* import = dialog->findChild<QPushButton*>("process_import_button");
	return new PVWidgets::PVDialogImportWidget(
	    dialog,
	    {.accept = import,
	     .buttons = {import, dialog->findChild<QPushButton*>("cancel_button")},
	     .sources = [this, dialog]() { return sources_of(*dialog); },
	     .set_paths = [dialog](QStringList const& paths) { dialog->add_captures(paths); },
	     .fills_page = true},
	    parent);
}

std::vector<PVWidgets::PVImportWidget::source_t>
PVPcapsicum::PVInputTypePcap::sources_of(ImportDialog const& dialog) const
{
	_input_paths = dialog.captures();

	// The format is written where the NRAW go, as is all that an import leaves behind.
	const QString format_dir =
	    PVCore::mkdtemp(PVRush::PVNrawCacheManager::nraw_dir() + "/pcap-format-XXXXXX");
	if (format_dir.isEmpty()) {
		throw std::runtime_error(QObject::tr("No temporary directory could be created in '%1'.")
		                             .arg(PVRush::PVNrawCacheManager::nraw_dir())
		                             .toStdString());
	}
	_tmp_dir_to_delete.insert(format_dir);
	const QDomDocument xml = dialog.format();
	const QString format_path =
	    format_dir + "/" + QString::number(QDateTime::currentMSecsSinceEpoch()) + ".format";
	QFile format_file(format_path);
	if (not format_file.open(QIODevice::WriteOnly)) {
		throw std::runtime_error(
		    QObject::tr("The format can't be written to '%1'.").arg(format_path).toStdString());
	}
	QTextStream text_stream(&format_file);
	xml.save(text_stream, 0);
	format_file.close();

	const pvpcap::extraction::options extraction = pvpcap::ws_extraction_options(dialog.profile());
	const auto import = std::make_shared<PVRush::PVPcapImport>(_input_paths.size());
	QList<PVRush::PVInputDescription_p> inputs;
	for (qsizetype i = 0; i < _input_paths.size(); i++) {
		inputs.push_back(PVRush::PVInputDescription_p(new PVRush::PVPcapDescription(
		    _input_paths[i], extraction, import, i, _input_paths.size() > 1)));
	}
	return {{inputs, PVRush::PVFormat(xml.documentElement())}};
}

PVWidgets::PVExporterWidgetInterface*
PVPcapsicum::PVInputTypePcap::create_exporter_widget(const list_inputs& inputs,
                                                     PVRush::PVNraw const& nraw) const
{
	return new PVGuiQt::PVPcapExporterWidget(inputs, nraw);
}

std::unique_ptr<PVRush::PVExporterBase>
PVPcapsicum::PVInputTypePcap::create_exporter(const list_inputs& inputs,
                                              PVRush::PVNraw const& nraw) const
{
	return std::make_unique<PVRush::PVPcapExporter>(inputs, nraw);
}

QString PVPcapsicum::PVInputTypePcap::get_exporter_filter_string(const list_inputs& inputs) const
{
	if (PVRush::PVPcapExporter::needs_pcapng(inputs)) {
		return ".pcapng files (*.pcapng)";
	} else {
		return ".pcap files (*.pcap)";
	}
}

QStringList PVPcapsicum::PVInputTypePcap::get_supported_extensions() const
{
	return {"pcap", "pcapng", "cap", "dmp"};
}

QString PVPcapsicum::PVInputTypePcap::name() const
{
	return {"pcap"};
}

QString PVPcapsicum::PVInputTypePcap::human_name() const
{
	return {"Pcap import plugin"};
}

QString PVPcapsicum::PVInputTypePcap::human_name_serialize() const
{
	return {"Pcap"};
}

QString PVPcapsicum::PVInputTypePcap::internal_name() const
{
	return {"04-pcap"};
}

QString PVPcapsicum::PVInputTypePcap::menu_input_name() const
{
	return {"Pcap"};
}

QString
PVPcapsicum::PVInputTypePcap::tab_name_of_inputs(PVRush::PVInputType::list_inputs const& in) const
{
	std::unordered_set<std::string> pcaps;
	for (const PVRush::PVInputDescription_p& input : in) {
		auto* f = dynamic_cast<PVRush::PVPcapDescription*>(input.get());
		pcaps.insert(QFileInfo(f->original_pcap_path()).fileName().toStdString());
	}
	std::vector<std::string> ordered_pcaps;
	ordered_pcaps.insert(ordered_pcaps.end(), pcaps.begin(), pcaps.end());
	std::sort(ordered_pcaps.begin(), ordered_pcaps.end());

	std::string tab_name = boost::algorithm::join(ordered_pcaps, "|");

	return QString::fromStdString(tab_name);
}

bool PVPcapsicum::PVInputTypePcap::get_custom_formats(PVRush::PVInputDescription_p /*in*/,
                                                      PVRush::hash_formats& /*formats*/) const
{
	return false;
}

QKeySequence PVPcapsicum::PVInputTypePcap::menu_shortcut() const
{
	return {};
}
