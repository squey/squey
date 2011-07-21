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
	PVCheckableComboBox(QString name, QWidget *parent = 0);

	/* functions */
	bool is_checked() const { return _checked; }
	void setChecked(bool checked);
	void addItems(QStringList items);

public slots:
	void checkStateChanged_Slot(int state);

};

} // namespace PVInspector

#endif	/* PVCHECKABLECOMBOBOX_H */
