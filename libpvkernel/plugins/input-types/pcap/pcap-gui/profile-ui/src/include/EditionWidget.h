#ifndef EDITIONWIDGET_H
#define EDITIONWIDGET_H

#include <QWidget>
#include <QString>

namespace Ui
{
class EditionWidget;
}

/**
 * It is the UI for monitoring job running process.
 */
class EditionWidget : public QWidget
{
	Q_OBJECT

  public:
	explicit EditionWidget(QWidget* parent = 0);
	~EditionWidget();

  public:
	QString selected_profile() const;

  Q_SIGNALS:
	void update_profile(QString profile);
	void profile_about_to_change(QString profile);

  private Q_SLOTS:
	void load_profile_list();
	void update_button_state();

	void on_new_profile_button_clicked();
	void on_delete_button_clicked();
	void on_duplicate_button_clicked();
	void on_rename_button_clicked();

  private:
	Ui::EditionWidget* _ui; //!< The ui generated interface.
};

#endif // EDITIONWIDGET_H
