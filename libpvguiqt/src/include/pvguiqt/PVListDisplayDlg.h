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

#ifndef PVGUIQT_PVLISTDISPLAYDLG_H
#define PVGUIQT_PVLISTDISPLAYDLG_H

#include "pvguiqt/ui_PVListDisplayDlg.h"

#include <pvkernel/core/PVArgument.h>
#include <pvguiqt/PVAbstractTableModel.h>

#include <QVector>
#include <QDialog>
#include <QFile>

namespace PVGuiQt
{

class PVLayerFilterProcessWidget;

class PVStatsSortProxyModel;

/**
 * This is the base class for all Widget with a "table" of value and
 * some options in it. It is use for example with invalid elements listing
 * or stat (distinct values, max values, ...) widgets.
 *
 * This class handle the model given in parameter (the one to display the
 * table of values) as we don't want to save the result of this computation
 * for a long time. Because of this constraint, the model should not have
 * parent otherwise, it will be double free.
 */
class PVListDisplayDlg : public QDialog, public Ui::PVListDisplayDlg
{
	Q_OBJECT

  public:
	explicit PVListDisplayDlg(PVAbstractTableModel* model, QWidget* parent = nullptr);

	/**
	 * Destructor to delete the underliying model.
	 */
	~PVListDisplayDlg() override;

  public:
	void set_description(QString const& desc);

  protected:
	virtual PVAbstractTableModel& model() { return *_model; }
	virtual PVAbstractTableModel const& model() const { return *_model; }

  protected:
	virtual void ask_for_copying_count() {}
	virtual bool process_context_menu(QAction* act);

  protected Q_SLOTS:
	/** Handle click on horizontal headers
	 *
	 * It sorts columns based on the clicked column but keep the current
	 * selection
	 *
	 * @param col : Index of the clicked column
	 */
	void copy_all_to_clipboard();
	void copy_selected_to_clipboard();
	void copy_to_file() { export_to_file_ui(false); }
	void append_to_file() { export_to_file_ui(true); }

	/**
	 * Show context menu for content listing.
	 */
	void show_ctxt_menu(const QPoint& pos);

  private:
	void export_to_file_ui(bool append);
	void export_to_file(QFile& file);
	/** Export count value in a QString
	 *
	 * Data can be extract from raw indices in the model.
	 *
	 * @param[in] count : Number of elements to extract
	 * @param[out] content : Exported line in a QString
	 * @return : Where it success or fail. It fails only in case of cancellation.
	 *
	 */
	bool export_values(int count, QString& content);

  protected:
	PVAbstractTableModel* _model;
	QAction* _copy_values_act;
	QMenu* _ctxt_menu;
};
} // namespace PVGuiQt

#endif
