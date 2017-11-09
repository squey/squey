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
