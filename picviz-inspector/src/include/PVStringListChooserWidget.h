//! \file PVStringListChooser.h
//! $Id: PVStringListChooserWidget.h 3168 2011-06-16 09:06:33Z aguinet $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011

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
