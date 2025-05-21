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

#include <pvkernel/core/PVArchive.h>
#include <pvkernel/core/PVDirectory.h>
#include <pvkernel/core/PVSpinBoxType.h>

#include "pcap/PVPcapDescription.h"
#include "pcap/PVPcapExporterWidget.h"

#include <QMessageBox>
#include <QFileInfo>
#include <QTextStream>
#include <QCheckBox>

#include <sys/time.h>
#ifndef _WIN32
#include <sys/resource.h>
#endif

PVPcapsicum::PVInputTypePcap::PVInputTypePcap()
    : PVRush::PVInputTypeDesc<PVRush::PVPcapDescription>()
{
	_limit_nfds = 1024;
#ifndef _WIN32
	struct rlimit rlim;
	if (getrlimit(RLIMIT_NOFILE, &rlim) == 0) {
		_limit_nfds =
		    rlim.rlim_cur - 1; // Take the soft limit as this is the one that will limit us...
	}
	else {
		PVLOG_WARN("Unable to get nofile limit. Uses 1024 by default.\n");
	}
#endif
}

PVPcapsicum::PVInputTypePcap::~PVInputTypePcap()
{
	for (const QString& tmp_dir : _tmp_dir_to_delete) {
		PVLOG_INFO("Delete temporary directory %s...\n", qPrintable(tmp_dir));
		PVCore::PVDirectory::remove_rec(tmp_dir);
	}
}

bool PVPcapsicum::PVInputTypePcap::load_files(pvpcap::splitted_files_t&& splitted_files,
                                              list_inputs& inputs,
                                              QWidget* parent) const
{
	std::map<std::string, std::pair<size_t, size_t>> pcap_offsets;
	for (pvpcap::splitted_file_t const& splitted_file : splitted_files) {
		std::pair<size_t, size_t>& v = pcap_offsets[splitted_file.original_pcap_path()];
		v.first += splitted_file.packets_count();
		v.second = splitted_file.streams_ids_count();
	}

	size_t packets_count_offset = 0;
	size_t streams_id_offset = 0;
	for (auto& pcount : pcap_offsets) {
		// packets count offsets
		size_t offset = pcount.second.first;
		pcount.second.first = packets_count_offset;
		packets_count_offset += offset;

		// streams id offsets
		offset = pcount.second.second;
		pcount.second.second = streams_id_offset;
		streams_id_offset += offset;
	}

	for (auto & filename : splitted_files) {
			inputs.push_back(PVRush::PVInputDescription_p(new PVRush::PVPcapDescription(
		    QString::fromStdString(filename.path()),
		    QString::fromStdString(filename.original_pcap_path()),
		    pcap_offsets[filename.original_pcap_path()].first, filename.packets_indexes(),
		    pcap_offsets[filename.original_pcap_path()].second, filename.streams_ids(),
		    _input_paths.size() > 1)));
	}

#ifdef __linux__
	if (inputs.size() >= _limit_nfds - 200) {
		ssize_t nopen = _limit_nfds - 200;
		if (nopen <= 0) {
			nopen = 1;
		}
		QString msg =
		    QObject::tr("You are trying to open %1 files, and your system limits a user to open %2 "
		                "file descriptor at once.\nConsidering the needs of the application, this "
		                "value must be set to a higher value. In order to change this limit, edit "
		                "/etc/security/limits.conf and add the following lines:")
		        .arg(inputs.size())
		        .arg(_limit_nfds);
		msg += "\n\n*\tsoft\tnofile\t131070\n*\thard\tnofile\t131070\n\n";
		msg += QObject::tr("You can set 131070 to a bigger value if needed. Then, you need to "
		                   "logout and login for these changes to be effectives.");
		msg += "\n\n";
		msg += QObject::tr("Only the first %1 file(s) will be opened.").arg(nopen);
		QMessageBox err(QMessageBox::Warning, QObject::tr("Too many files selected"), msg,
		                QMessageBox::Ok, parent);
		err.exec();
		inputs.erase(inputs.begin() + nopen + 1, inputs.end());
	}
#else
	(void) parent;
#endif

	return inputs.size() > 0;
}

bool PVPcapsicum::PVInputTypePcap::createWidget(PVRush::hash_formats& formats,
                                                PVRush::PVInputType::list_inputs& inputs,
                                                QString& format,
                                                PVCore::PVArgumentList& /*args_ext*/,
                                                QWidget* parent) const
{
	connect_parent(parent);

	std::unique_ptr<PVPcapParamsWidget> params(new PVPcapParamsWidget(parent));

	auto res = params->exec();

	_input_paths = params->pcap_paths();
	QString format_dir;
	for (pvpcap::splitted_file_t const& filename : params->csv_paths()) {
		const QString& dir_to_del =
		    QFileInfo(QString::fromStdString(filename.path())).dir().absolutePath();
		_tmp_dir_to_delete.insert(dir_to_del);
		if (format_dir.isEmpty()) {
			format_dir = dir_to_del;
		}
	}

	if (res == QDialog::Rejected or params->is_canceled()) {
		return false;
	}

	// store format in first temporary directory
	const QDomDocument& xml = params->get_format();
	QString format_path =
	    format_dir + "/" + QString::number(QDateTime::currentMSecsSinceEpoch()) + ".format";
	QFile format_file(format_path);
	format_file.open(QIODevice::WriteOnly);
	QTextStream text_stream(&format_file);
	xml.save(text_stream, 0);
	format_file.close();

	PVRush::PVFormat f(xml.documentElement());
	formats["custom"] = std::move(f);
	format = format_path;

	return load_files(params->csv_paths(), inputs, parent);
}

PVWidgets::PVExporterWidgetInterface*
PVPcapsicum::PVInputTypePcap::create_exporter_widget(const list_inputs& inputs,
                                                     PVRush::PVNraw const& nraw) const
{
	return new PVGuiQt::PVPcapExporterWidget(inputs, nraw);
}

/**
 * Returns true if all the input pcap files supports the pcapng format
 */
static bool is_pcapng_compatible(const PVRush::PVInputType::list_inputs& inputs)
{
	return std::all_of(inputs.begin(), inputs.end(), [](const auto& input_type_desc) {
		auto* fd =
		    dynamic_cast<PVRush::PVPcapDescription*>(input_type_desc.get());
		return (QFileInfo(fd->original_pcap_path()).suffix() == "pcapng");
	});
}

std::unique_ptr<PVRush::PVExporterBase>
PVPcapsicum::PVInputTypePcap::create_exporter(const list_inputs& inputs,
                                              PVRush::PVNraw const& nraw) const
{
	if (is_pcapng_compatible(inputs)) {
		return std::make_unique<PVRush::PVPcapNgExporter>(inputs, nraw);
	} else {
		return std::make_unique<PVRush::PVPcapExporter>(inputs, nraw);
	}
}

QString PVPcapsicum::PVInputTypePcap::get_exporter_filter_string(const list_inputs& inputs) const
{
	if (is_pcapng_compatible(inputs)) {
		return ".pcapng files (*.pcapng)";
	} else {
		return ".pcap files (*.pcap)";
	}
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
