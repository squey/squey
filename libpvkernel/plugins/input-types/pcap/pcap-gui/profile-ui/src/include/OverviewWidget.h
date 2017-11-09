#ifndef OVERVIEWWIDGET_H
#define OVERVIEWWIDGET_H

#include <QWidget>

#include <QString>
#include <QTreeWidget>

#include "rapidjson/document.h"

namespace Ui
{
class OverviewWidget;
}

/**
 * It is the UI for monitoring job running process.
 */
class OverviewWidget : public QWidget
{
	Q_OBJECT

  public:
	explicit OverviewWidget(const rapidjson::Document& json_data, QWidget* parent = 0);
	~OverviewWidget();

  public Q_SLOTS:
	/**
	 * Call this function when the model is updated.
	 */
	void update_model();

  private:
	void fill_tree_widget(const rapidjson::Value& value);
	void add_tree_root(QString name,
	                   QString filter,
	                   QString description,
	                   const rapidjson::Document& children);
	void add_tree_child(
	    QTreeWidgetItem* parent, QString name, QString filter, QString description, QString type);

  private:
	Ui::OverviewWidget* _ui;               //!< The ui generated interface.
	const rapidjson::Document& _json_data; //!< store profile JSON document.
};

#endif // OVERVIEWWIDGET_H
