#ifndef PVPRESETSWIDGET_H_
#define PVPRESETSWIDGET_H_

#include <QtGui>

namespace PVWidgets {

class PVPresetsWidget : public QWidget
{
	Q_OBJECT
public:
	friend class PVSavePresetAsDialog;

public:
	PVPresetsWidget(QWidget* parent = 0, Qt::WindowFlags f = 0);

public slots:
	void load_Slot();
	void save_Slot();
	void rename_Slot();
	void remove_Slot();
	void item_changed_Slot(QListWidgetItem* item);

Q_SIGNALS:
	void btn_load_clicked_Signal(const QString& preset);
	void btn_new_clicked_Signal(const QString& preset);
	void btn_save_clicked_Signal(const QString& preset);
	void btn_remove_clicked_Signal(const QString& preset);
	void preset_renamed_Signal(const QString& old_preset, const QString& new_preset);

public:
	void clear_presets();
	void add_presets(const QStringList& presets);
	void add_preset(const QString& preset, const QVariant& userData = QVariant());
	bool is_preset_txt_new(const QString& str) const;
	QVariant get_preset_data(int index) const;
	QVariant get_preset_data() const;
	QString get_current_preset_name() const;
	int get_preset_count() const;

private:
	QListWidget* _list;
	QPushButton* _btn_load;
	QPushButton* _btn_save;
	QPushButton* _btn_rename;
	QPushButton* _btn_remove;

	QString _last_preset_loaded;
	QString _old_preset_name;

};

class PVSavePresetAsDialog: public QDialog
{
	Q_OBJECT

public:
	friend class PVPresetsWidget;

public:
	PVSavePresetAsDialog(PVPresetsWidget* parent = 0, Qt::WindowFlags f = 0);

public slots:
	void save_Slot();
	void text_changed_Slot(const QString & text);

private:
	QComboBox* _comboBox;
	PVPresetsWidget* _parent;
	QPushButton* _btn_save;
};

}


#endif /* PVPRESETSWIDGET_H_ */
