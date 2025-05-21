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

#include "include/SelectionWidget.h"
#include "pcap-gui.h"

#include <QScreen>
#include <QGuiApplication>
#include <QMessageBox>
#include <QStandardPaths>

#include <pvkernel/widgets/PVFileDialog.h>

#include "ui_SelectionWidget.h"

#include "../include/libpvpcap.h"

class CloseEventFilter : public QObject
{
  public:
	explicit CloseEventFilter(ProfileWidget* profile_widget) : _profile_widget(profile_widget) {}

  protected:
	bool eventFilter(QObject* obj, QEvent* ev) override
	{
		if (ev->type() == QEvent::Close) {
			const QString& profile = _profile_widget->selected_profile();
			if (not profile.isEmpty()) {
				_profile_widget->ask_save_profile(profile);
			}
			ev->accept();
		}
		return QObject::eventFilter(obj, ev);
	}

  private:
	ProfileWidget* _profile_widget;
};

SelectionWidget::SelectionWidget(QWidget* parent) : QWidget(parent), _ui(new Ui::SelectionWidget)
{
	_ui->setupUi(this);

	// initialise profile list
	load_select_profile_combobox_list();
	update_select_pcap_list_button_state();

	// Manage profile dialog
	_profile_dialog = new QDialog(this);
	_profile_widget = new ProfileWidget(_profile_dialog);

	_profile_dialog->setLayout(new QVBoxLayout);
	_profile_dialog->layout()->addWidget(_profile_widget);

	QScreen* screen = QGuiApplication::primaryScreen();
	QRect screen_geo = screen->geometry();
	_profile_dialog->resize((double)screen_geo.width() / 1.2, (double)screen_geo.height() / 1.5);

	connect(_profile_widget, &ProfileWidget::closed, [&](QString selected_profile)
	{
		load_select_profile_combobox_list();
		if (not selected_profile.isEmpty()) {
			int index = _ui->select_profile_combobox->findText(selected_profile);
			if (index != -1) {
				_ui->select_profile_combobox->setCurrentIndex(index);
			}
		}
		_profile_dialog->close();
	});

	_profile_dialog->installEventFilter(new CloseEventFilter(_profile_widget));
}

SelectionWidget::~SelectionWidget()
{
	delete _ui;
}

void SelectionWidget::load_select_profile_combobox_list()
{
	_ui->select_profile_combobox->clear();
	for (auto& profile : pvpcap::get_system_profile_list()) {
		_ui->select_profile_combobox->addItem(
		    QString::fromStdString(profile),
		    QString::fromStdString(pvpcap::get_system_profile_dir()));
	}
	for (auto& profile : pvpcap::get_user_profile_list()) {
		_ui->select_profile_combobox->addItem(
		    QString::fromStdString(profile),
		    QString::fromStdString(pvpcap::get_user_profile_dir()));
	}
}

void SelectionWidget::on_add_button_clicked()
{
	QStringList filenames = PVWidgets::PVFileDialog::getOpenFileNames(
	    this, tr("Open PCAP files"), QStandardPaths::writableLocation(QStandardPaths::HomeLocation), tr("PCAP files (*.pcap *.pcapng)"));

	// Prevent input pcap files to be duplicated
	for (const QString& filename : filenames) {
		if (not _pcap_paths.contains(filename)) {
			_ui->select_pcap_list->addItem(filename);
			_pcap_paths.append(filename);
		}
	}

	update_select_pcap_list_button_state();
}

void SelectionWidget::on_remove_button_clicked()
{
	delete _ui->select_pcap_list->currentItem();
	_pcap_paths.clear();
	for (int i = 0; i < _ui->select_pcap_list->count(); ++i) {
		_pcap_paths.append(_ui->select_pcap_list->item(i)->text());
	}
	update_select_pcap_list_button_state();
}

void SelectionWidget::on_remove_all_button_clicked()
{
	_ui->select_pcap_list->clear();
	_pcap_paths.clear();
	update_select_pcap_list_button_state();
}

void SelectionWidget::update_select_pcap_list_button_state()
{
	// Enable button once file is added
	if (_ui->select_pcap_list->count() > 0) {
		_ui->remove_button->setEnabled(true);
		_ui->remove_all_button->setEnabled(true);
		_ui->process_import_button->setEnabled(true);
	} else {
		_ui->remove_button->setEnabled(false);
		_ui->remove_all_button->setEnabled(false);
		_ui->process_import_button->setEnabled(false);
	}
}
void SelectionWidget::on_cancel_button_clicked()
{
	Q_EMIT canceled();
}

void SelectionWidget::on_process_import_button_clicked()
{
	QString profile_path = _ui->select_profile_combobox->currentData().toString() +
	                       QDir::separator() + _ui->select_profile_combobox->currentText();

	// load json data from selected profile
	pvpcap::load_profile_data(_json_data, profile_path.toStdString());
	PVPcapsicum::check_wireshark_profile_exists(_json_data);

	auto* progess_dialog = new QDialog(this);
	_progress_widget = new ProgressWidget(_pcap_paths, get_tshark_cmd(), progess_dialog);
	connect(_progress_widget, &ProgressWidget::closed, progess_dialog, &QDialog::accept);

	progess_dialog->setLayout(new QVBoxLayout);
	progess_dialog->layout()->addWidget(_progress_widget);

	_progress_widget->run();

	progess_dialog->exec();

	Q_EMIT closed();
}

void SelectionWidget::on_manage_profile_button_clicked()
{
	_profile_widget->show();
	_profile_dialog->exec();
}
