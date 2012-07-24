/**
 * \file PVXmlParamComboBox.h
 *
 * Copyright (C) Picviz Labs 2011-2012
 */

#ifndef PVXMLPARAMCOMBOBOX_H
#define	PVXMLPARAMCOMBOBOX_H

#include <QComboBox>
#include <QString>
#include <QVariant>
#include <QStandardItemModel>

namespace PVInspector {

class PVXmlParamComboBox: public QComboBox {
    Q_OBJECT
public:
    PVXmlParamComboBox(QString name);
    virtual ~PVXmlParamComboBox();
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

#endif	/* PVXMLPARAMCOMBOBOX_H */
