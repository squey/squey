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
