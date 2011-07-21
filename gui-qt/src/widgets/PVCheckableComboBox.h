#ifndef PVCHECKABLECOMBOBOX_H
#define PVCHECKABLECOMBOBOX_H

#include <QWidget>
#include <QComboBox>
#include <QCheckBox>
#include <QLabel>

namespace PVInspector {

class PVCheckableComboBox : public QWidget
{
	Q_OBJECT
	/*  */
private:
	bool _checked;
	QCheckBox *checkbox;
	QComboBox *combobox;
	QLabel    *label;

public:
	PVCheckableComboBox(QWidget *parent = 0);

	/* functions */
	void addItems(QStringList items);
	void clear();
	int currentIndex() const { return combobox->currentIndex(); }
	bool is_checked() const { return _checked; }
	void setChecked(bool checked);
	void setCurrentIndex(int index) {combobox->setCurrentIndex(index);}
	void setText(QString text);

public slots:
	void checkStateChanged_Slot(int state);

};

} // namespace PVInspector

#endif	/* PVCHECKABLECOMBOBOX_H */
