#include "include/OptionWidget.h"
#include "ui_OptionWidget.h"

#include <QStringList>
#include <QFileDialog>

OptionWidget::OptionWidget(rapidjson::Document& json_data, QWidget* parent)
    : QWidget(parent), _ui(new Ui::OptionWidget), _json_data(json_data)
{
	_ui->setupUi(this);

	_ui->wireshark_filter_group->setVisible(false);
	_ui->wireshark_rewrite_ptions_group->setVisible(false);
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

void OptionWidget::on_filters_edit_textEdited(const QString& text)
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
		    QFileDialog::getExistingDirectory(this, tr("Open GeoIP Database directory"), "",
		                                      QFileDialog::ShowDirsOnly)
		        .toStdString();
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
