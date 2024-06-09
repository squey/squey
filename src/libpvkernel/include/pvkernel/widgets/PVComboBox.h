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

#ifndef PVWIDGETS_PVCOMBOBOX_H
#define PVWIDGETS_PVCOMBOBOX_H

#include <qcontainerfwd.h>
#include <qlist.h>
#include <qnamespace.h>
#include <qstring.h>
#include <qvariant.h>
#include <QComboBox>
#include <QStandardItemModel>
#include <QWidget>

class QModelIndex;
class QObject;
class QWidget;

namespace PVWidgets
{

class PVComboBox : public QComboBox
{
  public:
	explicit PVComboBox(QWidget* parent);

  public:
	QString get_selected() const;
	QVariant get_sel_userdata() const;
	bool select(QString const& str);
	bool select_userdata(QVariant const& data);

  public:
	// Disabled strings handling
	void add_disabled_string(QString const& str);
	void remove_disabled_string(QString const& str);
	void clear_disabled_strings();
	inline QStringList& disabled_strings() { return _dis_elt; }
	const QStringList& disabled_strings() const { return _dis_elt; }

  protected:
	QStringList _dis_elt;

  protected:
	// This model allows for items to be disabled inside the combo box
	class PVComboBoxModel : public QStandardItemModel
	{
	  public:
		explicit PVComboBoxModel(QStringList& dis_elt, QObject* parent = nullptr);
		Qt::ItemFlags flags(const QModelIndex& index) const override;
		QVariant data(const QModelIndex& index, int role) const override;

	  protected:
		bool is_disabled(const QModelIndex& index) const;

	  protected:
		QStringList& _dis_elt;
	};
};
} // namespace PVWidgets

#endif
