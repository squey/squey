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

#ifndef PROFILEWIDGET_H
#define PROFILEWIDGET_H

#include <QWidget>

#include <profile-ui/src/include/EditionWidget.h>
#include <profile-ui/src/include/TreeWidget.h>
#include <profile-ui/src/include/OverviewWidget.h>
#include <profile-ui/src/include/OptionWidget.h>

#include "../include/libpvpcap/profileformat.h"
#include "../include/libpvpcap/ws.h"

namespace Ui
{
class ProfileWidget;
}

/**
 * It is the UI for monitoring job running process.
 */
class ProfileWidget : public QWidget
{
	Q_OBJECT

  public:
	explicit ProfileWidget(QWidget* parent = 0);
	~ProfileWidget();

  public:
	void load_profile_data(const QString& profile);
	void update_view_state();
	QString selected_profile();

  Q_SIGNALS:
	void closed();
	void propagate_update_profile();

  public Q_SLOTS:
	void ask_save_profile(QString profile);

  private Q_SLOTS:
	void on_cancel_button_clicked();
	void on_close_button_clicked();
	void on_save_button_clicked();

  private:
	std::string _profile_name;          //!< store profile name.
	rapidjson::Document _json_data;     //!< store profile JSON document.
	rapidjson::Document _json_overview; //!< store profile JSON document.

	Ui::ProfileWidget* _ui; //!< The ui generated interface.
	EditionWidget* _edition_widget;
	TreeWidget* _tree_widget;
	OverviewWidget* _overview_widget;
	OptionWidget* _option_widget;
};

#endif // PROFILEWIDGET_H
