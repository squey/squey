#ifndef PVPRESETSWIDGET_H_
#define PVPRESETSWIDGET_H_

#include <QtGui>

namespace PVInspector {

class PVPresetsWidget : public QWidget
{
	Q_OBJECT
public:
	PVPresetsWidget(QWidget* parent = 0, Qt::WindowFlags f = 0);

public slots:
	void new_Slot();
	void load_Slot();
	void save_Slot();
	void remove_Slot();
	void text_changed_Slot(const QString& text);

Q_SIGNALS:
	void btn_load_clicked_Signal(const QString& preset);
	void btn_new_clicked_Signal(const QString& preset);
	void btn_save_clicked_Signal(const QString& preset);
	void btn_remove_clicked_Signal(const QString& preset);

public:
	void clear_presets();
	void add_presets(const QStringList& presets);
	void add_preset(const QString& preset, const QVariant& userData = QVariant());
	bool is_preset_txt_new();
	QVariant get_preset_data(int index);
	QVariant get_preset_data();
	QString get_current_preset_name();
	int get_preset_count();

private:
	QHBoxLayout* _layout;
	QLabel*      _label;
	QComboBox*   _combo;
	QPushButton* _btn_new;
	QPushButton* _btn_load;
	QPushButton* _btn_save;
	QPushButton* _btn_remove;
};
}


#endif /* PVPRESETSWIDGET_H_ */
