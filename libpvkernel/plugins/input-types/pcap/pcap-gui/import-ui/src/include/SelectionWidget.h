#ifndef SELECTIONWIDGET_H
#define SELECTIONWIDGET_H

#include <QDialog>
#include <QFileDialog>
#include <QVBoxLayout>
#include <QWidget>

#include <import-ui/src/include/ProgressWidget.h>
#include <profile-ui/src/include/ProfileWidget.h>

namespace Ui
{
class SelectionWidget;
}

/**
 * It is the UI for monitoring job running process.
 */
class SelectionWidget : public QWidget
{
	Q_OBJECT
  public:
	explicit SelectionWidget(QWidget* parent = 0);
	~SelectionWidget();

  public:
	std::vector<std::string> get_tshark_cmd() { return pvpcap::ws_get_cmdline_opts(_json_data); }

	QStringList pcap_paths() const { return _pcap_paths; }

	pvpcap::splitted_files_t csv_paths() const
	{
		return _progress_widget ? _progress_widget->csv_paths() : pvpcap::splitted_files_t();
	}
	QDomDocument get_format() const { return pvpcap::get_format(_json_data, _pcap_paths.size()); }

	bool is_canceled() const { return _progress_widget->is_canceled(); }

  Q_SIGNALS:
	void closed();
	void canceled();
	void process_import_button_clicked();

  private Q_SLOTS:
	void on_add_button_clicked();
	void on_remove_button_clicked();
	void on_remove_all_button_clicked();
	void on_cancel_button_clicked();
	void on_manage_profile_button_clicked();
	void on_process_import_button_clicked();

	void update_select_pcap_list_button_state();
	void load_select_profile_combobox_list();

  private:
	rapidjson::Document _json_data; //!< store profile JSON document.
	Ui::SelectionWidget* _ui;       //!< The ui generated interface.
	QStringList _pcap_paths;

	ProfileWidget* _profile_widget;
	QDialog* _profile_dialog;

	ProgressWidget* _progress_widget = nullptr;
};

#endif // SELECTIONWIDGET_H
