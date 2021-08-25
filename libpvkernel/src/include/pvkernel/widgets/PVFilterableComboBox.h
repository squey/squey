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

#ifndef __PVFILTERABLECOMBOBOX_H__
#define __PVFILTERABLECOMBOBOX_H__

#include <QCompleter>
#include <QStringListModel>
#include <QSortFilterProxyModel>

namespace PVWidgets
{

class PVFilterableComboBox : public QComboBox
{
  public:
	PVFilterableComboBox(QWidget* parent = nullptr) : QComboBox(parent)
	{
		setEditable(true);

		// setup model
		_proxy_model.setSourceModel(&_list_model);
		_proxy_model.setFilterCaseSensitivity(Qt::CaseInsensitive);
		setModel(&_proxy_model);

		// setup completion
		QCompleter* completer = new QCompleter(&_proxy_model, this);
		completer->setCompletionMode(QCompleter::UnfilteredPopupCompletion);
		setCompleter(completer);
		connect(lineEdit(), &QLineEdit::textChanged, this, &PVFilterableComboBox::on_text_changed);
		connect(completer,
		        static_cast<void (QCompleter::*)(const QString&)>(&QCompleter::activated), this,
		        &PVFilterableComboBox::on_completer_activated);
	}

  public:
	void set_string_list(const QStringList& string_list) { _list_model.setStringList(string_list); }

  private:
	void on_text_changed()
	{
		const QStringList& Items = _list_model.stringList();
		const QString& txt = lineEdit()->text();
		for (const QString& item : Items) {
			if (item.indexOf(txt) > -1) {
				_proxy_model.setFilterFixedString(txt);
				return;
			}
		}
	}

	void on_completer_activated(const QString& index_name)
	{
		if (not index_name.isEmpty()) {
			setCurrentText(index_name);
		}
	}

  private:
	QStringListModel _list_model;
	QSortFilterProxyModel _proxy_model;
};

} // namespace PVWidgets

#endif // __PVFILTERABLECOMBOBOX_H__
