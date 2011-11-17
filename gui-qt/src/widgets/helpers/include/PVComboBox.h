#ifndef WIDGETS_PVCOMBOBOX_H
#define WIDGETS_PVCOMBOBOX_H

#include <QComboBox>
#include <QStandardItemModel>
#include <QWidget>

namespace PVInspector {

namespace PVWidgetsHelpers {

class PVComboBox: public QComboBox {
public:
	PVComboBox(QWidget* parent);

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
	class PVComboBoxModel: public QStandardItemModel {
	public:
		PVComboBoxModel(QStringList& dis_elt, QObject* parent = 0);
		virtual Qt::ItemFlags flags(const QModelIndex &index) const;
		QVariant data(const QModelIndex& index, int role) const;
	protected:
		bool is_disabled(const QModelIndex& index) const;
	protected:
		QStringList& _dis_elt;
	};
};

}

}

#endif
