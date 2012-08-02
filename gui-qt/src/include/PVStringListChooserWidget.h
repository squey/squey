/**
 * \file PVStringListChooserWidget.h
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#include <QWidget>
#include <QDialog>
#include <QStringList>
#include <QListWidget>

namespace PVInspector {

class PVStringListChooserWidget : public QDialog
{
	Q_OBJECT

public:
	PVStringListChooserWidget(QWidget *parent_, QString const& text, QStringList const& list, QStringList comments = QStringList());
public:
	QStringList get_sel_list();
public slots:
	void ok_Slot();
protected:
	QListWidget* _list_w;
	QStringList  _final_list;
};

}
