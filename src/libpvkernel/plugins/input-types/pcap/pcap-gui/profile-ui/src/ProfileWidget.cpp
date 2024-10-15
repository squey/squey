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

#include "include/ProfileWidget.h"
#include "ui_ProfileWidget.h"

#include <profile-ui/models/include/ProtocolFieldListModel.h>

#include "pcap/libpvpcap/include/libpvpcap.h"
#include "pcap/libpvpcap/include/libpvpcap/ws.h"
#include "pcap/libpvpcap/include/libpvpcap/profileformat.h"

#include <QMessageBox>

#include <iostream>

ProfileWidget::ProfileWidget(QWidget* parent)
    : QWidget(parent)
    , _ui(new Ui::ProfileWidget)
    , _edition_widget(new EditionWidget)
    , _tree_widget(new TreeWidget(&_json_data))
    , _overview_widget(new OverviewWidget(_json_data))
    , _option_widget(new OptionWidget(_json_data))
{
	_ui->setupUi(this);
	_ui->edition_box->layout()->addWidget(_edition_widget);
	_ui->tree_tab->layout()->addWidget(_tree_widget);
	_ui->overview_tab->layout()->addWidget(_overview_widget);
	_ui->options_tab->layout()->addWidget(_option_widget);

	connect(_edition_widget, &EditionWidget::update_profile, this,
	        &ProfileWidget::load_profile_data);

	connect(this, &ProfileWidget::propagate_update_profile, _tree_widget,
	        &TreeWidget::update_model);

	connect(this, &ProfileWidget::propagate_update_profile, _overview_widget,
	        &OverviewWidget::update_model);

	connect(this, &ProfileWidget::propagate_update_profile, _option_widget,
	        &OptionWidget::load_option_from_json);

	connect(_tree_widget, &TreeWidget::update_tree_data, _overview_widget,
	        &OverviewWidget::update_model);

	connect(_tree_widget, &TreeWidget::propagate_selection, _overview_widget,
	        &OverviewWidget::update_model);

	connect(_edition_widget, &EditionWidget::profile_about_to_change, this,
	        &ProfileWidget::ask_save_profile);

	update_view_state();
}

ProfileWidget::~ProfileWidget()
{
	delete _ui;
	delete _edition_widget;
	delete _tree_widget;
	delete _overview_widget;
	delete _option_widget;
}

void ProfileWidget::ask_save_profile(QString profile)
{
	const std::string& profile_path = pvpcap::get_user_profile_dir() + "/" + profile.toStdString();

	if (not pvpcap::file_exists(pvpcap::get_user_profile_path(profile.toStdString()))) {
		return; // profile was deleted
	}

	rapidjson::Document previous_profile_content;
	pvpcap::load_profile_data(previous_profile_content, profile_path);

	if (previous_profile_content != _json_data) {
		if (QMessageBox::question(this, "Save profile ?",
		                          QString("The profile '%1' contains unsaved modifications.\n"
		                                  "Do you want to save them?")
		                              .arg(profile),
		                          QMessageBox::Yes | QMessageBox::No) == QMessageBox::Yes) {
			pvpcap::save_profile_data(_json_data, profile_path);
		}
	}
}

QString ProfileWidget::selected_profile()
{
	return _edition_widget->selected_profile();
}

void ProfileWidget::on_cancel_button_clicked()
{
	close();
	Q_EMIT closed("");
}

void ProfileWidget::on_close_button_clicked()
{
	QString profile = selected_profile();
	close();
	Q_EMIT closed(profile);
}

void ProfileWidget::on_save_button_clicked()
{
	const std::string& profile_path = pvpcap::get_user_profile_dir() + "/" + _profile_name;
	pvpcap::save_profile_data(_json_data, profile_path);
}

void ProfileWidget::load_profile_data(const QString& profile)
{
	const std::string& profile_path = pvpcap::get_user_profile_dir() + "/" + profile.toStdString();

	_profile_name = profile.toStdString();
	pvpcap::load_profile_data(_json_data, profile_path);
	Q_EMIT propagate_update_profile();
	update_view_state();
}

void ProfileWidget::update_view_state()
{
	bool read_only = true;
	if (not _json_data.IsNull())
		read_only = _json_data["read_only"].GetBool();

	_ui->save_button->setEnabled(not read_only);
	_tree_widget->setEnabled(not read_only);
	_overview_widget->setEnabled(not read_only);
	_option_widget->setEnabled(not read_only);
}
