#include <PVAxesCombinationDialog.h>
#include <PVAxesCombinationWidget.h>
#include <PVMainWindow.h>
#include <PVListingView.h>

#include <QVBoxLayout>

PVInspector::PVAxesCombinationDialog::PVAxesCombinationDialog(Picviz::PVView_sp view, PVTabSplitter* tab_, PVMainWindow* mw):
	QDialog(mw),
	_view(view)
{
	main_window = mw;
	tab = tab_;

	QVBoxLayout* main_layout = new QVBoxLayout();
	QDialogButtonBox* box_buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
	_axes_widget = new PVAxesCombinationWidget(view->axes_combination, mw, view.get());
	main_layout->addWidget(_axes_widget);
	main_layout->addWidget(box_buttons);
	setLayout(main_layout);

	// Buttons
	connect(box_buttons, SIGNAL(accepted()), tab_, SLOT(source_changed_Slot()));
	connect(box_buttons, SIGNAL(accepted()), this, SLOT(accept()));
	connect(box_buttons, SIGNAL(rejected()), this, SLOT(cancel_slot()));

	// Axes combination widget
	connect(_axes_widget, SIGNAL(axes_combination_changed()), this, SLOT(refresh_axes_slot())); 
	connect(_axes_widget, SIGNAL(axes_count_changed()), this, SLOT(axes_count_changed_slot())); 

	setWindowTitle("Edit axes combination...");
}

void PVInspector::PVAxesCombinationDialog::refresh_axes_slot()
{
	// AG: this is a hack because adding a new column into the axes combination
	// actually breaks the selection model of the listing (thus it isn't possible to
	// get the actual selected rows).
	// TOFIX: use the axis combination in the proxy model and not in the original model !
	QItemSelectionModel* sel_model = tab->get_listing_view()->selectionModel();
	QModelIndexList sel_idxes = sel_model->selectedIndexes();
	QSet<PVRow> sel_lines;
	sel_lines.reserve(sel_idxes.size()/_view->axes_combination.get_original_axes_count());
	foreach(QModelIndex const& idx, sel_idxes) {
		sel_lines << idx.row();
	}

	tab->refresh_listing_with_horizontal_header_Slot();
	tab->update_pv_listing_model_Slot();
	tab->refresh_listing_Slot();
	main_window->update_pvglview(_view, PVSDK_MESSENGER_REFRESH_POSITIONS);

	// Clear the selection and set it
	sel_model->clearSelection();
	foreach(PVRow r, sel_lines) {
		sel_model->select(sel_model->model()->index(r,0), QItemSelectionModel::Rows | QItemSelectionModel::Select);
	}
}

void PVInspector::PVAxesCombinationDialog::axes_count_changed_slot()
{
	main_window->update_pvglview(_view, PVSDK_MESSENGER_REFRESH_AXES_COUNT);
}

void PVInspector::PVAxesCombinationDialog::cancel_slot()
{
	_axes_widget->restore_saved_combination();
	reject();
}

void PVInspector::PVAxesCombinationDialog::save_current_combination()
{
	_axes_widget->save_current_combination();
}

void PVInspector::PVAxesCombinationDialog::update_used_axes()
{
	_axes_widget->update_used_axes();
}
