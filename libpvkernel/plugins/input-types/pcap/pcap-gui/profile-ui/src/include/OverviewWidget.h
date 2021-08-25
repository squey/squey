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
