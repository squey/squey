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

#ifndef __PVFILTERABLEMENU_H__
#define __PVFILTERABLEMENU_H__

#include <QMenu>
#include <QLineEdit>
#include <QWidgetAction>
#include <QCompleter>
#include <QStringListModel>
#include <QSortFilterProxyModel>

static constexpr const int MAX_ITEM_COUNT = 30;

namespace PVWidgets
{

class PVFilterableMenu : public QMenu
{
  public:
	PVFilterableMenu(QWidget* parent = nullptr) : PVFilterableMenu("", parent) {}

	PVFilterableMenu(const QString& title, QWidget* parent = nullptr) : QMenu(parent)
	{
		setTitle(title);

		// Enable scrolling
		setStyleSheet("QMenu { menu-scrollable: 1; }");

		// Setup model
		_proxy_model.setSourceModel(&_list_model);
		_proxy_model.setFilterCaseSensitivity(Qt::CaseInsensitive);
	}

  public:
	void addActions(QList<QAction*> actions)
	{
		// Enable filtering only when there is too many options in the menu
		if (actions.size() >= MAX_ITEM_COUNT) {
			_search_edit = new QLineEdit(this);

			QCompleter* completer = new QCompleter(&_proxy_model, this);
			completer->setCompletionMode(QCompleter::PopupCompletion);
			completer->setCaseSensitivity(Qt::CaseInsensitive);
			completer->setFilterMode(Qt::MatchContains);
			completer->setModelSorting(QCompleter::CaseInsensitivelySortedModel);
			_search_edit->setCompleter(completer);
			connect(_search_edit, &QLineEdit::textChanged, this,
			        &PVFilterableMenu::on_text_changed);
			connect(completer,
			        static_cast<void (QCompleter::*)(const QString&)>(&QCompleter::activated), this,
			        &PVFilterableMenu::on_completer_activated);
			QWidgetAction* search_act = new QWidgetAction(_search_edit);
			search_act->setDefaultWidget(_search_edit);
			addAction(search_act);
		}

		// Add actions
		QStringList string_list;
		for (const QAction* act : actions) {
			string_list << act->text();
		}
		_list_model.setStringList(string_list);

		QMenu::addActions(actions);
	}

  private:
	void on_text_changed()
	{
		const QStringList& items = _list_model.stringList();
		const QString& txt = _search_edit->text();
		for (const QString& item : items) {
			if (item.indexOf(txt) > -1) {
				_proxy_model.setFilterRegularExpression(txt);
				return;
			}
		}
	}

	void on_completer_activated(const QString& action_text)
	{
		if (not action_text.isEmpty()) {
			for (QAction* action : actions()) {
				if (action->text() == action_text) {
					_search_edit->clear();
					action->trigger();
					break;
				}
			}
		}
	}

  private:
	QLineEdit* _search_edit = nullptr;
	QStringListModel _list_model;
	QSortFilterProxyModel _proxy_model;
};

} // namespace PVWidgets

#endif // __PVFILTERABLEMENU_H__
