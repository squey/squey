#ifndef OPTIONWIDGET_H
#define OPTIONWIDGET_H

#include <QWidget>

#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"

namespace Ui
{
class OptionWidget;
}

/**
 * It is the UI for monitoring job running process.
 */
class OptionWidget : public QWidget
{
	Q_OBJECT

  public:
	explicit OptionWidget(rapidjson::Document& json_data, QWidget* parent = 0);
	~OptionWidget();

	void load_option_from_json();

  private Q_SLOTS:
	void on_source_check_clicked(bool checked = false);
	void on_destination_check_clicked(bool checked = false);
	void on_protocol_check_clicked(bool checked = false);
	void on_info_check_clicked(bool checked = false);

	void on_ws_profiles_combobox_currentTextChanged(const QString& text);

	// TCP/IP
	void on_reassemble_stream_checkbox_clicked(bool checked = false);
	void on_defragment_ip_streams_checkbox_clicked(bool checked = false);

	// Name resolution
	void on_resolve_ip_checkbox_checkbox_clicked(bool checked = false);
	void on_use_pcap_for_address_resolution_clicked(bool checked = false);
	void on_query_dn_server_checkbox_clicked(bool checked = false);
	void on_geoip_db_button_clicked(bool checked = false);

	void on_filters_edit_textEdited(const QString& text);

	void on_header_check_clicked(bool checked = false);

	void on_separator_edit_textEdited(const QString& text);
	void on_occurrence_edit_textEdited(const QString& text);
	void on_aggregator_edit_textEdited(const QString& text);
	void on_quote_edit_textEdited(const QString& text);

  private:
	Ui::OptionWidget* _ui;           //!< The ui generated interface.
	rapidjson::Document& _json_data; //!< store profile JSON document.
};

#endif // OPTIONWIDGET_H
