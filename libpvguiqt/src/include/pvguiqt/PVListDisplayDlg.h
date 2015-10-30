/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVGUIQT_PVLISTDISPLAYDLG_H
#define PVGUIQT_PVLISTDISPLAYDLG_H

#include <pvguiqt/ui_PVListDisplayDlg.h>

#include <pvhive/PVActor.h>
#include <pvhive/PVObserverSignal.h>

#include <picviz/PVView_types.h>

#include <pvkernel/core/PVArgument.h>

#include <QAbstractListModel>
#include <QVector>
#include <QDialog>
#include <QFileDialog>

namespace PVGuiQt {

class PVLayerFilterProcessWidget;

class PVStringSortProxyModel;

class PVListDisplayDlg: public QDialog, public Ui::PVListDisplayDlg
{
	Q_OBJECT

public:
	PVListDisplayDlg(QAbstractListModel* model,  QWidget* parent = NULL);

public:
	void set_description(QString const& desc);

protected:
	QAbstractListModel* model();
	PVStringSortProxyModel* proxy_model();

protected:
	virtual void ask_for_copying_count() {}
	virtual void sort_by_column(int col);
	virtual bool process_context_menu(QAction* act);

	/** Export a line in a QString format
	 *
	 * Extract the model index for the i-th elements using f and return its
	 * formated content
	 *
	 * @param[in] model: The model containing data
	 * @param[in] f : Funtion to extract the index in the model from global index
	 * @param[in] i : Global index to extract
	 * @return : Qstring content of the line
	 */
	virtual QString export_line(
		PVGuiQt::PVStringSortProxyModel* model,
		std::function<QModelIndex(int)> f,
		int i
	);

protected slots:
	/** Handle click on horizontal headers
	 *
	 * It sorts columns based on the clicked column but keep the current
	 * selection
	 *
	 * @param col : Index of the clicked column
	 */
	void section_clicked(int col);
	void copy_all_to_clipboard();
	void copy_selected_to_clipboard();
	void copy_to_file() { export_to_file_ui(false); }
	void append_to_file() { export_to_file_ui(true); }
	void sort();
	void show_ctxt_menu(const QPoint& pos);

private:
	void export_to_file_ui(bool append);
	void export_to_file(QFile& file);
	/** Export count value in a QString
	 *
	 * Data can be extract from raw indices in the model but also from anything
	 * as long as the access is provided through the f function.
	 *
	 * @param[in] count : Number of elements to extract
	 * @param[in] f : Function to find the i-th elements.
	 * @param[out] content : Exported line in a QString
	 * @return : Where it success or fail. It fails only in case of cancellation.
	 *
	 */
	bool export_values(int count, std::function<QModelIndex (int)> f, QString& content);

protected:
	QFileDialog _file_dlg;
	QAction* _copy_values_act;
	QMenu* _ctxt_menu;
	PVGuiQt::PVLayerFilterProcessWidget* _ctxt_process = nullptr;
	PVCore::PVArgumentList _ctxt_args;
	//QItemSelection _item_selection;
};

}

#endif
