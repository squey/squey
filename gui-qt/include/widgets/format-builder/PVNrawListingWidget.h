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

#ifndef PVNRAWLISTINGWIDGET_H
#define PVNRAWLISTINGWIDGET_H

#include <pvkernel/rush/PVInputType.h>
#include <pvkernel/rush/PVInputDescription.h>

#include <pvguiqt/PVLocalizedSpinBox.h>

#include <QWidget>
#include <QSpinBox>
#include <QPushButton>
#include <QLabel>
#include <QHeaderView>
#include <QTableView>

namespace PVInspector
{

// Forward declaration
class PVNrawListingModel;

class PVNrawListingWidget : public QWidget
{
	Q_OBJECT
  public:
	PVNrawListingWidget(PVNrawListingModel* nraw_model, QWidget* parent = nullptr);

  public:
	template <typename T, typename F>
	void connect_preview(T* receiver, const F& slot)
	{
		connect(_btn_preview, &QPushButton::clicked, receiver, slot);
	}

	template <typename T, typename F>
	void connect_autodetect(T* receiver, const F& slot)
	{
		connect(_btn_autodetect, &QPushButton::clicked, receiver, slot);
	}

	template <typename T, typename F>
	void connect_axes_name(T* receiver, const F& slot)
	{
		connect(this, &PVNrawListingWidget::set_axes_name_from_nraw, receiver, slot);
	}

	template <typename T, typename F>
	void connect_axes_type(T* receiver, const F& slot)
	{
		connect(this, &PVNrawListingWidget::set_axes_type_from_nraw, receiver, slot);
	}

	template <typename T, typename F>
	void connect_table_header(T* receiver, const F& slot)
	{
		connect(_nraw_table->horizontalHeader(), &QHeaderView::sectionClicked,
		        [=](int c) { (receiver->*slot)(PVCol(c)); });
	}

	void get_ext_args(PVRow& start, PVRow& end);
	void get_autodetect_args(PVRow& start, PVRow& end);
	void set_autodetect_count(PVRow count);
	void set_last_input(PVRush::PVInputType_p in_t = PVRush::PVInputType_p(),
	                    PVRush::PVInputDescription_p input = PVRush::PVInputDescription_p());
	void resize_columns_content();
	void unselect_column();
	void select_column(PVCol col);
	void mark_row_as_invalid(size_t row_index);
	void select_header(PVCol column);

  public Q_SLOTS:
	void nraw_custom_menu_Slot(const QPoint& pt);
	void set_axes_name_selected_row_Slot();
	void set_axes_type_selected_row_Slot();

  Q_SIGNALS:
	void set_axes_name_from_nraw(int row);
	void set_axes_type_from_nraw(int row);

  protected:
	int get_selected_row();

  protected:
	PVNrawListingModel* _nraw_model;
	PVGuiQt::PVLocalizedSpinBox* _ext_start;
	PVGuiQt::PVLocalizedSpinBox* _ext_count;
	QPushButton* _btn_preview;
	PVGuiQt::PVLocalizedSpinBox* _autodetect_start;
	PVGuiQt::PVLocalizedSpinBox* _autodetect_count;
	QPushButton* _btn_autodetect;
	QLabel* _src_label;
	QTableView* _nraw_table;
	QMenu* _ctxt_menu;
};
} // namespace PVInspector

#endif
