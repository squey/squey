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

#include "include/OptionWidget.h"
#include "ui_OptionWidget.h"
#include "pcap-gui.h"
#include <libpvpcap/ws.h>

#include <QStringList>
#include <QStandardPaths>

#include <unistd.h>
#include <sys/types.h>
#include <sys/types.h>
#include <sys/stat.h>

#include <pvkernel/widgets/PVFileDialog.h>

OptionWidget::OptionWidget(rapidjson::Document& json_data, QWidget* parent)
    : QWidget(parent), _ui(new Ui::OptionWidget), _json_data(json_data)
{
	_ui->setupUi(this);

	_ui->wireshark_rewrite_ptions_group->setVisible(false);

	// Populate wireshark profiles combo box
	for (const std::string& profile_path : pvpcap::get_wireshark_profiles_paths()) {
		_ui->ws_profiles_combobox->addItem(
		    QFileInfo(QString::fromStdString(profile_path)).fileName());
	}
	_ui->ws_profiles_combobox->setCurrentText("Default");

	connect(_ui->filters_edit, &QPlainTextEdit::textChanged, [this](){ on_filters_edit_textChanged(_ui->filters_edit->toPlainText());});
}

OptionWidget::~OptionWidget()
{
	delete _ui;
}

void OptionWidget::load_option_from_json()
{
	if (not _json_data.IsNull()) {
		rapidjson::Value& options = _json_data["options"];

		_ui->source_check->setChecked(options["source"].GetBool());
		_ui->destination_check->setChecked(options["destination"].GetBool());
		_ui->protocol_check->setChecked(options["protocol"].GetBool());
		_ui->info_check->setChecked(options["info"].GetBool());

		_ui->occurrence_edit->setText(options["occurrence"].GetString());
		_ui->aggregator_edit->setText(options["aggregator"].GetString());

		_ui->filters_edit->setPlainText(options["filters"].GetString());

		// TCP/IP
		if (options.HasMember("tcp.desegment_tcp_streams")) {
			_ui->reassemble_stream_checkbox->setChecked(
			    options["tcp.desegment_tcp_streams"].GetBool());
		}
		if (options.HasMember("ip.defragment")) {
			_ui->defragment_ip_streams_checkbox->setChecked(options["ip.defragment"].GetBool());
		}

		// Name resolution
		if (options.HasMember("nameres.network_name")) {
			_ui->resolve_ip_checkbox_checkbox->setChecked(
			    options["nameres.network_name"].GetBool());
		}
		if (options.HasMember("nameres.dns_pkt_addr_resolution")) {
			_ui->use_pcap_for_address_resolution->setChecked(
			    options["nameres.dns_pkt_addr_resolution"].GetBool());
		}
		if (options.HasMember("nameres.use_external_name_resolver")) {
			_ui->query_dn_server_checkbox->setChecked(
			    options["nameres.use_external_name_resolver"].GetBool());
		}
		if (options.HasMember("geoip_db_paths")) {
			std::string dirname = options["geoip_db_paths"].GetString();
			QString path_url = QString("<a href=\"file://") + QString::fromStdString(dirname) +
			                   QString("\">") + QString::fromStdString(dirname) + QString("</a>");
			_ui->geoip_db_path_label->setText(path_url);
		}
		if (options.HasMember("wireshark_profile")) {
			const std::string& wireshark_profile = options["wireshark_profile"].GetString();
			_ui->ws_profiles_combobox->setCurrentText(QString::fromStdString(wireshark_profile));
			if (_ui->ws_profiles_combobox->currentText() !=
			    QString::fromStdString(wireshark_profile)) {
				PVPcapsicum::check_wireshark_profile_exists(_json_data);
			}
		} else {
			_ui->ws_profiles_combobox->setCurrentText("Default");
		}
	}
}

void OptionWidget::on_source_check_clicked(bool checked)
{
	if (not _json_data.IsNull()) {
		_json_data["options"]["source"].SetBool(checked);
	}
}

void OptionWidget::on_destination_check_clicked(bool checked)
{
	if (not _json_data.IsNull()) {
		_json_data["options"]["destination"].SetBool(checked);
	}
}

void OptionWidget::on_protocol_check_clicked(bool checked)
{
	if (not _json_data.IsNull()) {
		_json_data["options"]["protocol"].SetBool(checked);
	}
}

void OptionWidget::on_info_check_clicked(bool checked)
{
	if (not _json_data.IsNull()) {
		_json_data["options"]["info"].SetBool(checked);
	}
}

void OptionWidget::on_header_check_clicked(bool checked)
{
	if (not _json_data.IsNull()) {
		_json_data["options"]["header"].SetBool(checked);
	}
}

void OptionWidget::on_filters_edit_textChanged(const QString& text)
{
	// Fixme: validate filters string regexp

	if (not _json_data.IsNull()) {
		_json_data["options"]["filters"].SetString(text.toStdString().c_str(),
		                                           _json_data.GetAllocator());
	}
}

void OptionWidget::on_separator_edit_textEdited(const QString& text)
{
	// Fixme: validate string regexp

	if (not _json_data.IsNull()) {
		_json_data["options"]["separator"].SetString(text.toStdString().c_str(),
		                                             _json_data.GetAllocator());
	}
}

void OptionWidget::on_occurrence_edit_textEdited(const QString& text)
{
	// Fixme: validate string regexp

	if (not _json_data.IsNull()) {
		_json_data["options"]["occurrence"].SetString(text.toStdString().c_str(),
		                                              _json_data.GetAllocator());
	}
}

void OptionWidget::on_aggregator_edit_textEdited(const QString& text)
{
	// Fixme: validate string regexp

	if (not _json_data.IsNull()) {
		_json_data["options"]["aggregator"].SetString(text.toStdString().c_str(),
		                                              _json_data.GetAllocator());
	}
}

void OptionWidget::on_quote_edit_textEdited(const QString& text)
{
	// Fixme: validate string regexp

	if (not _json_data.IsNull()) {
		_json_data["options"]["quote"].SetString(text.toStdString().c_str(),
		                                         _json_data.GetAllocator());
	}
}

void OptionWidget::on_reassemble_stream_checkbox_clicked(bool checked /*= false*/)
{
	if (not _json_data.IsNull()) {
		if (not _json_data["options"].HasMember("tcp.desegment_tcp_streams")) {
			rapidjson::Document::AllocatorType& alloc = _json_data.GetAllocator();
			rapidjson::Value val;
			val.SetBool(checked);
			_json_data["options"].AddMember("tcp.desegment_tcp_streams", val, alloc);
		} else {
			_json_data["options"]["tcp.desegment_tcp_streams"].SetBool(checked);
		}
	}
}

void OptionWidget::on_defragment_ip_streams_checkbox_clicked(bool checked /*= false*/)
{
	if (not _json_data.IsNull()) {
		if (not _json_data["options"].HasMember("ip.defragment")) {
			rapidjson::Document::AllocatorType& alloc = _json_data.GetAllocator();
			rapidjson::Value val;
			val.SetBool(checked);
			_json_data["options"].AddMember("ip.defragment", val, alloc);
		} else {
			_json_data["options"]["ip.defragment"].SetBool(checked);
		}
	}
}

void OptionWidget::on_resolve_ip_checkbox_checkbox_clicked(bool checked /*= false*/)
{
	if (not _json_data.IsNull()) {
		if (not _json_data["options"].HasMember("nameres.network_name")) {
			rapidjson::Document::AllocatorType& alloc = _json_data.GetAllocator();
			rapidjson::Value val;
			val.SetBool(checked);
			_json_data["options"].AddMember("nameres.network_name", val, alloc);
		} else {
			_json_data["options"]["nameres.network_name"].SetBool(checked);
		}
	}
}

void OptionWidget::on_use_pcap_for_address_resolution_clicked(bool checked /*= false*/)
{
	if (not _json_data.IsNull()) {
		if (not _json_data["options"].HasMember("nameres.dns_pkt_addr_resolution")) {
			rapidjson::Document::AllocatorType& alloc = _json_data.GetAllocator();
			rapidjson::Value val;
			val.SetBool(checked);
			_json_data["options"].AddMember("nameres.dns_pkt_addr_resolution", val, alloc);
		} else {
			_json_data["options"]["nameres.dns_pkt_addr_resolution"].SetBool(checked);
		}
	}
}

void OptionWidget::on_query_dn_server_checkbox_clicked(bool checked /*= false*/)
{
	if (not _json_data.IsNull()) {
		if (not _json_data["options"].HasMember("nameres.use_external_name_resolver")) {
			rapidjson::Document::AllocatorType& alloc = _json_data.GetAllocator();
			rapidjson::Value val;
			val.SetBool(checked);
			_json_data["options"].AddMember("nameres.use_external_name_resolver", val, alloc);
		} else {
			_json_data["options"]["nameres.use_external_name_resolver"].SetBool(checked);
		}
	}
}

void OptionWidget::on_geoip_db_button_clicked(bool /*checked = false*/)
{
	if (not _json_data.IsNull()) {
		std::string dirname =
		    PVWidgets::PVFileDialog::getExistingDirectory(
				this,
				tr("Open GeoIP Database directory"),
		        QStandardPaths::writableLocation(QStandardPaths::HomeLocation),
				QFileDialog::ShowDirsOnly).toStdString();
		if (not dirname.empty()) {
			if (not _json_data["options"].HasMember("geoip_db_paths")) {
				rapidjson::Document::AllocatorType& alloc = _json_data.GetAllocator();
				rapidjson::Value val;
				val.SetString(dirname.c_str(), dirname.size(), _json_data.GetAllocator());
				_json_data["options"].AddMember("geoip_db_paths", val, alloc);
			} else {
				_json_data["options"]["geoip_db_paths"].SetString(dirname.c_str(), dirname.size(),
				                                                  _json_data.GetAllocator());
			}
			QString path_url = QString("<a href=\"file://") + QString::fromStdString(dirname) +
			                   QString("\">") + QString::fromStdString(dirname) + QString("</a>");
			_ui->geoip_db_path_label->setText(path_url);
		}
	}
}

void OptionWidget::on_ws_profiles_combobox_currentTextChanged(const QString& text)
{
	if (not _json_data.IsNull()) {
		if (not _json_data["options"].HasMember("wireshark_profile") and text != "Default") {
			rapidjson::Document::AllocatorType& alloc = _json_data.GetAllocator();
			rapidjson::Value val;
			val.SetString(text.toStdString().c_str(), text.size(), _json_data.GetAllocator());
			_json_data["options"].AddMember("wireshark_profile", val, alloc);
		} else {
			if (text != "Default") {
				_json_data["options"]["wireshark_profile"].SetString(
				    text.toStdString().c_str(), text.size(), _json_data.GetAllocator());
			} else {
				_json_data["options"].RemoveMember("wireshark_profile");
			}
		}
	}
}
