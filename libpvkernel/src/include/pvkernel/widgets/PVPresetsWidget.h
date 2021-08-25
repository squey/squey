/* * MIT License
 *
 * © ESI Group, 2015
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef PVPRESETSWIDGET_H_
#define PVPRESETSWIDGET_H_

#include <pvkernel/core/PVLogger.h>

#include <QGroupBox>
#include <QDialog>
#include <QComboBox>
#include <QHBoxLayout>
#include <QAction>
#include <QMessageBox>
#include <QPushButton>
#include <QLabel>
#include <QKeyEvent>
#include <QToolBar>
#include <QListWidget>

namespace PVWidgets
{

namespace __impl
{
class PVPresetsListWidget;
} // namespace __impl

class PVPresetsWidget : public QWidget
{
	Q_OBJECT
  public:
	friend class PVSavePresetAsDialog;

  public:
	explicit PVPresetsWidget(const QString& title,
	                         QWidget* parent = nullptr);

  public Q_SLOTS:
	void load_Slot();
	void save_Slot();
	void rename_Slot();
	void remove_Slot();
	void item_changed_Slot(QListWidgetItem* item);
	void update_actions_availability();

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
	void select_preset(int idx);

  private:
	QGroupBox* _group_box;
	__impl::PVPresetsListWidget* _list;
	QToolBar* _toolbar;

	QAction* _loadAct;
	QAction* _saveAct;
	QAction* _renameAct;
	QAction* _removeAct;

	QString _last_preset_loaded;
	QString _old_preset_name;
};

namespace __impl
{

class PVPresetsListWidget : public QListWidget
{
  public:
	explicit PVPresetsListWidget(PVWidgets::PVPresetsWidget* parent) : QListWidget((QWidget*)parent)
	{
		_parent = parent;
	}
	void keyPressEvent(QKeyEvent* event) override
	{
		if (event->key() == Qt::Key_F2) {
			_parent->rename_Slot();
		} else if (event->key() == Qt::Key_Delete) {
			_parent->remove_Slot();
		}
	}

	QSize sizeHint() const override { return QSize(0, 42); }

  private:
	PVWidgets::PVPresetsWidget* _parent;
};
} // namespace __impl

class PVSavePresetAsDialog : public QDialog
{
	Q_OBJECT

  public:
	friend class PVPresetsWidget;

  public:
	explicit PVSavePresetAsDialog(PVPresetsWidget* parent = nullptr);

  public Q_SLOTS:
	void save_Slot();
	void text_changed_Slot(const QString& text);

  private:
	QComboBox* _comboBox;
	PVPresetsWidget* _parent;
	QPushButton* _btn_save;
};
} // namespace PVWidgets

#endif /* PVPRESETSWIDGET_H_ */
