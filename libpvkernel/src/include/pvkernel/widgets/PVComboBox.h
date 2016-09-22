/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVWIDGETS_PVCOMBOBOX_H
#define PVWIDGETS_PVCOMBOBOX_H

#include <QComboBox>
#include <QStandardItemModel>
#include <QWidget>

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
