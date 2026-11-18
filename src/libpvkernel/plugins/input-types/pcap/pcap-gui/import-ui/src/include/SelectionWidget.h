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

#ifndef SELECTIONWIDGET_H
#define SELECTIONWIDGET_H

#include <QDialog>
#include <QDomDocument>
#include <QVBoxLayout>
#include <QWidget>

#include <profile-ui/src/include/ProfileWidget.h>

#include "rapidjson/document.h"

namespace Ui
{
class SelectionWidget;
}

/**
 * Where the captures to import and the profile to read them with are chosen.
 */
class SelectionWidget : public QWidget
{
	Q_OBJECT
  public:
	explicit SelectionWidget(QWidget* parent = 0);
	~SelectionWidget();

  public:
	QStringList pcap_paths() const { return _pcap_paths; }

	//! The profile chosen, once the import was asked for.
	const rapidjson::Document& profile() const { return _json_data; }

	QDomDocument get_format() const;

	void add_files(const QStringList& pcap_paths);

  Q_SIGNALS:
	void closed();
	void canceled();

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
};

#endif // SELECTIONWIDGET_H
