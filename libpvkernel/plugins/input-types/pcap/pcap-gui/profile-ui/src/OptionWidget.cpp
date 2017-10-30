#include "include/OptionWidget.h"
#include "ui_OptionWidget.h"

#include <QStringList>

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
