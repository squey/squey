//! \file PVFilterWidget.cpp
//! $Id: PVFilterWidget.cpp 2496 2011-04-25 14:10:00Z psaade $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <QtCore>
#include <QtGui>

#include <QVBoxLayout>
#include <QDialog>
#include <QMessageBox>
#include <QFrame>
#include <QVBoxLayout>
#include <QLabel>
#include <QStringList>

#include <PVMainWindow.h>
#include <PVDualSlider.h>

#include <pvkernel/core/general.h>
//FIXME #include <picviz/filters.h>
#include <picviz/arguments.h>
#include <picviz/PVSource.h>
#include <picviz/PVStateMachine.h>
#include <picviz/PVView.h>

#include <pvkernel/rush/PVFormat.h>

#include <PVFilterWidget.h>

/******************************************************************************
 *
 * PVInspector::PVFilterWidget::PVFilterWidget
 *
 *****************************************************************************/
PVInspector::PVFilterWidget::PVFilterWidget(PVInspector::PVMainWindow *parent) : QDialog(parent)
{
	main_window = parent;

	//VARIABLES
	QFrame      *separator      = new QFrame;
	QPushButton *cancel         = new QPushButton("Cancel");
	QPushButton *apply          = new QPushButton("Apply");
	QPushButton *ok             = new QPushButton("OK");
	QVBoxLayout *main_layout    = new QVBoxLayout;
	QHBoxLayout *buttons_layout = new QHBoxLayout;
	filter_widgets_layout = new QGridLayout;

	//CODE
	filter_widgets_layout->setSizeConstraint(QLayout::SetFixedSize);
	main_layout->addLayout(filter_widgets_layout);

	// // We add a separator
	separator->setFrameShape(QFrame::HLine);
	separator->setFrameShadow(QFrame::Raised);
	main_layout->addWidget(separator);

	// We add the apply button
	main_layout->addLayout(buttons_layout);

	buttons_layout->addWidget(cancel);
	connect(cancel, SIGNAL(pressed()), this, SLOT(filter_cancel_action_Slot()));

	apply->setDefault(true);
	connect(apply, SIGNAL(pressed()), this, SLOT(filter_apply_action_Slot()));
	buttons_layout->addWidget(apply);

	buttons_layout->addWidget(ok);
	connect(ok, SIGNAL(pressed()), this, SLOT(filter_ok_action_Slot()));

	setLayout(main_layout);
	setWindowTitle(QString("Run filter ..."));
}

/******************************************************************************
 *
 * PVInspector::PVFilterWidget::create
 *
 *****************************************************************************/
#if 0 // FIXME!
void PVInspector::PVFilterWidget::create(QString filter_name_, picviz_filter_t *filter_)
{
	picviz_arguments_t *arguments;
	int i = 0;
	QLayoutItem *item;

	//	qDebug("filter name='%s'", filter_name.toUtf8().data());

	current_tab = main_window->current_tab;

	//delete main_layout;
	//main_layout = new QVBoxLayout;
	while ((item = filter_widgets_layout->takeAt(0))) {
		item->widget()->hide();
		delete item;
	}
	
	filter_name = strdup(filter_name_.toUtf8().data()); // XXX is there any reason for not using a QString here?

	//this->filter_widgets_layout = new QGridLayout;
	filter = filter_;

	if (!current_tab) {
		PVLOG_ERROR("Current view does not exists. Cannot apply any filter!\n");
		return;
	}

	arguments = filter->get_arguments_func();
	if (arguments) {

// 		filter_widgets_layout->setSizeConstraint(QLayout::SetFixedSize);

		for (i=0; i<arguments->nelts; i++) {
		  if (arguments->args[i].widget == PICVIZ_ARGUMENT_WIDGET_DUALSLIDER) {
					// FIXME: We need en ENSURE the dual slider is properly built in the plugin (two consecutive arguments being part of the group etc...)
					PVDualSlider *pv_DualSlider = new PVDualSlider(0); // XXX Ahem This constructor is probably sick!

					pv_DualSlider->setObjectName(arguments->args[i].group);

					filter_widgets_layout->setColumnStretch(1, 1);
					filter_widgets_layout->setColumnMinimumWidth(1, 250);
					filter_widgets_layout->setRowMinimumHeight(i, 50);

					filter_widgets_layout->addWidget(pv_DualSlider, i, 1);
					pv_DualSlider->show();

					i++; // We do a second jump because a dual sliders needs to have consecutive arguments
					continue;
		  }
		  if (arguments->args[i].widget == PICVIZ_ARGUMENT_WIDGET_TEXTBOX) {
					QLabel *label = new QLabel(arguments->args[i].name);
					QLineEdit *lineedit = new QLineEdit();
				
					lineedit->setObjectName(arguments->args[i].name);

					filter_widgets_layout->addWidget(label, i, 1);
					label->show();
					filter_widgets_layout->addWidget(lineedit, i, 2);
					lineedit->show();
		  }
		  if (arguments->args[i].widget == PICVIZ_ARGUMENT_WIDGET_AXIS_CHOOSER) {
		  	QComboBox *combo_axes_list = new QComboBox();
			QStringList axes_list;
			QLabel *axes_label = new QLabel(arguments->args[i].name);

		  	picviz_source_t *source;
			PVRush::PVFormat *format;

			source = picviz_view_get_source_parent(current_tab->get_lib_view());
			format = source->format;
			// axes_list = picviz_format_get_axes_name(format);
			// QString axes(picviz_format_get_axes_name(format));
			for (int cur_i = 0; cur_i < format->axes.size(); ++cur_i) {
				axes_list << format->axes[cur_i]["name"];
			}			
			combo_axes_list->addItems(axes_list);

			combo_axes_list->setObjectName(arguments->args[i].name);

			filter_widgets_layout->addWidget(axes_label, i, 1);
			axes_label->show();

			filter_widgets_layout->addWidget(combo_axes_list, i, 2);
			combo_axes_list->show();

		  }
		}	  // for
// 		main_layout->addLayout(filter_widgets_layout);

		show();

	} else { 		// Run filters that need no arguments
		char *filter_error;
		/* We do what has to be done in the lib FIRST */
		filter_error = picviz_view_apply_filter_from_name(current_tab->get_lib_view(), filter_name, filter->get_arguments_func());
		if (!filter_error) {
			picviz_view_process_from_eventline(current_tab->get_lib_view());
			main_window->update_pvglview(current_tab->get_lib_view(), PVGL_COM_REFRESH_COLOR|PVGL_COM_REFRESH_SELECTION);
		} else {
			QMessageBox msgBox;
			msgBox.setIcon(QMessageBox::Critical);
			msgBox.setText(filter_error);
			msgBox.exec();
		}
	}

	emit filter_applied_Signal();
}
#endif

/******************************************************************************
 *
 * PVInspector::PVFilterWidget::filter_exec_action_build_arguments
 *
 *****************************************************************************/
picviz_arguments_t *PVInspector::PVFilterWidget::filter_exec_action_build_arguments(void)
{
	PVLOG_DEBUG("PVInspector::PVFilterWidget::%s\n", __FUNCTION__);

	// QObject *s = sender();
	// char *sender_name = strdup(s->objectName().toUtf8().data());
	picviz_arguments_t *arguments = 0;
#if 0 // FIXME!
	picviz_argument_item_t item;

	int nb_rows;
	int itempos;

	PVDualSlider *pv_DualSlider;
	QString group;
	char *group_text;

	QComboBox *combobox;
	QLineEdit *lineedit;
	QLabel *label;
	char *lineedit_text;
	char *label_text;

	// Widgets number: one per line, so we count rows from the QDialog->filter_widgets_layout (QGridLayout)
	nb_rows = filter_widgets_layout->rowCount();
	arguments = filter->get_arguments_func();

	for (itempos=0; itempos<arguments->nelts; itempos++) {
		switch(arguments->args[itempos].widget) {
			case PICVIZ_ARGUMENT_WIDGET_DUALSLIDER:
				pv_DualSlider = (PVDualSlider *)filter_widgets_layout->itemAtPosition(itempos, 1)->widget();
				group = pv_DualSlider->objectName();
				group_text = group.toUtf8().data();

				/* We fetch and set the value of the left slider */
				item = picviz_arguments_get_item_from_group_and_dualslider_position(arguments, group_text, PICVIZ_ARGUMENT_SLIDER_LEFT);
				item.fval = pv_DualSlider->get_slider_value(0);
				picviz_arguments_set_item_from_name(arguments, item.name, item);

				/* We fetch and set the value of the right slider */
				item = picviz_arguments_get_item_from_group_and_dualslider_position(arguments, group_text, PICVIZ_ARGUMENT_SLIDER_RIGHT);
				item.fval = pv_DualSlider->get_slider_value(1);
				picviz_arguments_set_item_from_name(arguments, item.name, item);

				itempos++;
				break;
			case PICVIZ_ARGUMENT_WIDGET_TEXTBOX:
				lineedit = (QLineEdit *)filter_widgets_layout->itemAtPosition(itempos, 2)->widget();
				lineedit_text = strdup(lineedit->text().toUtf8().data());

				label = (QLabel *)filter_widgets_layout->itemAtPosition(itempos, 1)->widget();
				label_text = strdup(label->text().toUtf8().data());

				PVLOG_DEBUG("We can set the property '%s' with value '%s' to the filter named '%s'\n", label_text, lineedit_text, filter_name);

				item = picviz_arguments_get_item_from_name(arguments, label_text);
				item = picviz_arguments_item_set_string(item, lineedit_text);
				picviz_arguments_set_item_from_name(arguments, item.name, item);

				free(lineedit_text);
				free(label_text);
				break;
			case PICVIZ_ARGUMENT_WIDGET_AXIS_CHOOSER:
				combobox = (QComboBox *)filter_widgets_layout->itemAtPosition(itempos, 2)->widget();

				label = (QLabel *)filter_widgets_layout->itemAtPosition(itempos, 1)->widget();
				label_text = strdup(label->text().toUtf8().data());
				
				item = picviz_arguments_get_item_from_name(arguments, label_text);
				item.ival = combobox->currentIndex();
				picviz_arguments_set_item_from_name(arguments, item.name, item);

				free(label_text);

				break;
			default:
			  printf("ERROR: No such widget!\n");
			  break;
		
		} // switch
	}   // for (i=0; i<arguments->nelts; i++)

	// picviz_view_apply_filter_from_name(current_tab->get_lib_view(), filter_name, arguments);
	// picviz_view_process_from_eventline(current_tab->get_lib_view());

	// //hide();

	// current_tab->refresh_view_Slot(); XXX ???

	// emit filter_applied_Signal();
#endif
	return arguments;
}

/******************************************************************************
 *
 * PVInspector::PVFilterWidget::filter_cancel_action_Slot
 *
 *****************************************************************************/
void PVInspector::PVFilterWidget::filter_cancel_action_Slot()
{
#if 0 // FIXME!
	picviz_arguments_t *arguments;

	arguments = filter_exec_action_build_arguments();

	// picviz_view_apply_filter_from_name(current_tab->get_lib_view(), filter_name, arguments);
	picviz_view_process_from_selection(current_tab->get_lib_view());

	hide();

	main_window->update_pvglview(current_tab->get_lib_view(), PVGL_COM_REFRESH_COLOR|PVGL_COM_REFRESH_SELECTION);

	emit filter_applied_Signal();
#endif
}

/******************************************************************************
 *
 * PVInspector::PVFilterWidget::filter_apply_action_Slot
 *
 *****************************************************************************/
void PVInspector::PVFilterWidget::filter_apply_action_Slot()
{
#if 0 // FIXME
	picviz_arguments_t *arguments;

	arguments = filter_exec_action_build_arguments();

	picviz_view_apply_filter_from_name(current_tab->get_lib_view(), filter_name, arguments);
	picviz_view_process_from_eventline(current_tab->get_lib_view());

	main_window->update_pvglview(current_tab->get_lib_view(), PVGL_COM_REFRESH_COLOR|PVGL_COM_REFRESH_SELECTION);

	emit filter_applied_Signal();
#endif
}



/******************************************************************************
 *
 * PVInspector::PVFilterWidget::filter_ok_action_Slot
 *
 *****************************************************************************/
void PVInspector::PVFilterWidget::filter_ok_action_Slot()
{
#if 0 // FIXME
	picviz_arguments_t *arguments;
	picviz_view_t *lib_view = current_tab->get_lib_view();

	arguments = filter_exec_action_build_arguments();


	picviz_view_apply_filter_from_name(lib_view, filter_name, arguments);
	//picviz_view_process_from_eventline(current_tab->get_lib_view());

	// We use the post_filter_layer result as the new state for layer_stack_output_layer
	// Warning : this will be destroyed by any further action INVOLVING a reprocess of the layer stack
	// Commit if you want to keep this for further use.
	picviz_layer_A2B_copy(lib_view->post_filter_layer, lib_view->pre_filter_layer);
	picviz_selection_A2B_copy(lib_view->post_filter_layer->selection, lib_view->volatile_selection);
	lib_view->state_machine->set_square_area_mode(Picviz::PVStateMachine::AREA_MODE_SET_WITH_VOLATILE);

	// We reprocess the pipeline from the eventline stage
	picviz_view_process_from_eventline(lib_view);

	//OLD
	// picviz_selection_A2B_copy(lib_view->real_output_selection, lib_view->volatile_selection);
	// picviz_selection_A2B_copy(lib_view->real_output_selection, lib_view->pre_filter_layer->selection);

	// picviz_view_process_from_layer_stack(lib_view);
	// // picviz_selection_A2B_copy(lib_view->real_output_selection, lib_view->pre_filter_layer->selection);
	// picviz_layer_A2B_copy(lib_view->post_filter_layer, lib_view->layer_stack_output_layer);
	// picviz_selection_A2A_select_all(lib_view->real_output_selection);
	// picviz_state_machine_set_square_area_mode(lib_view->state_machine, PICVIZ_SM_SQUARE_AREA_MODE_SET_WITH_VOLATILE);
	// picviz_view_process_from_selection(lib_view);

	hide();

	main_window->update_pvglview(current_tab->get_lib_view(), PVGL_COM_REFRESH_COLOR|PVGL_COM_REFRESH_SELECTION);

	emit filter_applied_Signal();
#endif
}

// // This slot receives from the sender name
// // the group name that allows to retrieve the widget
// // and their associated values to then launch the plugin execution
// /******************************************************************************
//  *
//  * PVMainWindow::dualslider_value_changed_Slot
//  *
//  *****************************************************************************/
// void PVMainWindow::dualslider_value_changed_Slot(void)
// {
// 	/* VARIABLES */
// 	QObject *s = sender();
// 	QString str;
// 	picviz_arguments_t *args;
// 	picviz_argument_item_t item;
// 	char *sender_name = strdup(s->objectName().toUtf8().data());

// 	/* CODE */
// 	args = this->filter->get_arguments_func();

// 	/* We fetch and set the value of the first argument */
// 	item = picviz_arguments_get_item_from_group_and_dualslider_position(args, sender_name, PICVIZ_ARGUMENT_SLIDER_LEFT);
// 	item.fval = pv_DualSlider->get_slider_value(0);
// 	picviz_arguments_set_item_from_name(args, item.name, item);

// 	/* We fetch and set the value of the second argument */
// 	item = picviz_arguments_get_item_from_group_and_dualslider_position(args, sender_name, PICVIZ_ARGUMENT_SLIDER_RIGHT);
// 	item.fval = pv_DualSlider->get_slider_value(1);
// 	picviz_arguments_set_item_from_name(args, item.name, item);


// 	//PVLOG_INFO("0=%f, 1=%f\n", pv_DualSlider->get_slider_value(0), pv_DualSlider->get_slider_value(1));
// 	/* We process the filter */
// 	picviz_view_apply_filter_from_name(current_tab->get_lib_view(), this->last_sendername, args);

// 	/* We process the view from the eventline */
// 	picviz_view_process_from_eventline(current_tab->get_lib_view());

// 	/* We free the memory */
// 	item = picviz_arguments_get_item_from_group_and_dualslider_position(args, sender_name, PICVIZ_ARGUMENT_SLIDER_LEFT);
// 	picviz_arguments_item_destroy(item);
// 	item = picviz_arguments_get_item_from_group_and_dualslider_position(args, sender_name, PICVIZ_ARGUMENT_SLIDER_RIGHT);
// 	picviz_arguments_item_destroy(item);

// 	free(sender_name);

// 	/* THEN we can emit the signal */
// 	emit filter_applied_Signal();
// }


