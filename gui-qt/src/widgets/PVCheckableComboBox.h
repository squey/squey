/**
 * \file PVCheckableComboBox.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVCHECKABLECOMBOBOX_H
#define PVCHECKABLECOMBOBOX_H

#include <QWidget>
#include <QComboBox>
#include <QCheckBox>
#include <QLabel>

#include <QMouseEvent>

#include <pvkernel/core/general.h>

namespace PVInspector {

class PVCheckableComboBox : public QWidget
{
	Q_OBJECT
	/* Q_PROPERTY(int _current_index READ currentIndex WRITE setCurrentIndex USER true) */
	/*  */
private:
	bool _checked;
	int _current_index;

	QCheckBox *checkbox;
	QComboBox *combobox;
	QLabel    *label;

public:
	PVCheckableComboBox(QWidget *parent = 0);

	/* functions */
	void addItems(QStringList items);
	void clear();
	/* int currentIndex() const { PVLOG_INFO("WE GET CURRENTINDEX()\n"); return combobox->currentIndex(); } */
	int currentIndex() const { PVLOG_INFO("GRAB CURRENT INDEX\n"); return _current_index; }
	/* bool eventFilter(QObject *o, QEvent *e); */
	bool is_checked() const { PVLOG_INFO("WE GET IS_CHECKED\n"); return _checked; }
	void setChecked(bool checked);
	void setCurrentIndex(int index) {PVLOG_INFO("WE SET THE CURRENT INDEX FROM OUR WIDGET\n"); combobox->setCurrentIndex(index);}
	void setText(QString text);

public slots:
	void checkStateChanged_Slot(int state);
	void comboIndexChanged_Slot(int index);

protected:
	void mouseReleaseEvent(QMouseEvent *event);

};

} // namespace PVInspector

#endif	/* PVCHECKABLECOMBOBOX_H */
