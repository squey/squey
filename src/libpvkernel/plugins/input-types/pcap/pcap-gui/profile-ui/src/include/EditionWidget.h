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
