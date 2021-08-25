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

#ifndef PVXMLPARAMCOMBOBOX_H
#define PVXMLPARAMCOMBOBOX_H

#include <QComboBox>
#include <QString>
#include <QVariant>
#include <QStandardItemModel>

namespace PVInspector
{

class PVXmlParamComboBox : public QComboBox
{
	Q_OBJECT
  public:
	PVXmlParamComboBox(QString name);
	~PVXmlParamComboBox() override;
	QVariant val();
	void select(QString const& title);
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
		PVComboBoxModel(QStringList& dis_elt, QObject* parent = nullptr);
		Qt::ItemFlags flags(const QModelIndex& index) const override;
		QVariant data(const QModelIndex& index, int role) const override;

	  protected:
		bool is_disabled(const QModelIndex& index) const;

	  protected:
		QStringList& _dis_elt;
	};
};
} // namespace PVInspector

#endif /* PVXMLPARAMCOMBOBOX_H */
