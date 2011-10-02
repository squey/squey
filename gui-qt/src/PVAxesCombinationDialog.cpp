#include <PVAxesCombinationDialog.h>
#include <PVAxesCombinationWidget.h>
#include <PVMainWindow.h>

#include <QVBoxLayout>

PVInspector::PVAxesCombinationDialog::PVAxesCombinationDialog(PVTabSplitter* tab_, PVMainWindow* mw):
	QDialog(mw)
{
	main_window = mw;
	tab = tab_;

	QVBoxLayout* main_layout = new QVBoxLayout();
	QDialogButtonBox* box_buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
	_axes_widget = new PVAxesCombinationWidget(tab->get_lib_view()->axes_combination, mw);
	main_layout->addWidget(_axes_widget);
	main_layout->addWidget(box_buttons);
	setLayout(main_layout);

	// Buttons
	connect(box_buttons, SIGNAL(accepted()), this, SLOT(accept()));
	connect(box_buttons, SIGNAL(rejected()), this, SLOT(cancel_slot()));

	// Axes combination widget
	connect(_axes_widget, SIGNAL(axes_combination_changed()), this, SLOT(refresh_axes_slot())); 
	connect(_axes_widget, SIGNAL(axes_count_changed()), this, SLOT(axes_count_changed_slot())); 

	setWindowTitle("Edit axes combination...");
}

void PVInspector::PVAxesCombinationDialog::refresh_axes_slot()
{
	tab->refresh_listing_with_horizontal_header_Slot();
	tab->update_pv_listing_model_Slot();
	tab->refresh_listing_Slot();
	main_window->update_pvglview(tab->get_lib_view(), PVGL_COM_REFRESH_POSITIONS);
}

void PVInspector::PVAxesCombinationDialog::axes_count_changed_slot()
{
	main_window->update_pvglview(tab->get_lib_view(), PVGL_COM_REFRESH_AXES_COUNT);
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
