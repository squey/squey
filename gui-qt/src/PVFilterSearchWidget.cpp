//! \file PVFilterSearchWidget.cpp
//! $Id: PVFilterSearchWidget.cpp 3090 2011-06-09 04:59:46Z stricaud $
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

#include <pvcore/general.h>

#include <pvcore/debug.h>
//FIXME #include <picviz/filters.h>

#include <pvrush/PVFormat.h>

#include <picviz/arguments.h>
#include <picviz/PVSource.h>
#include <picviz/PVView.h>
#include <picviz/state-machine.h>

#include <pvrush/PVFormat.h>

#include <PVFilterSearchWidget.h>

/******************************************************************************
 *
 * PVInspector::PVFilterSearchWidget::PVFilterSearchWidget
 *
 *****************************************************************************/
PVInspector::PVFilterSearchWidget::PVFilterSearchWidget(PVInspector::PVMainWindow *parent, QString const& regexp, int axis_id, bool show_regexp) : QDialog(parent)
{
	main_window = parent;

	//VARIABLES
	QFrame      *separator      = new QFrame;
	QPushButton *cancel         = new QPushButton("Cancel");
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

	buttons_layout->addWidget(ok);
	connect(ok, SIGNAL(pressed()), this, SLOT(search_filter_ok_action_Slot()));

	buttons_layout->addWidget(cancel);
	connect(cancel, SIGNAL(pressed()), this, SLOT(search_filter_cancel_action_Slot()));



	// Add the label search with its textbox
	QLabel *label = new QLabel("Search: ");
	searchline_edit = new QLineEdit();
	searchline_edit->setText(regexp);
	
	int cur_line = 0;
	if (show_regexp) {
		filter_widgets_layout->addWidget(label, cur_line, 1);
		filter_widgets_layout->addWidget(searchline_edit, cur_line, 2);
		label->show();
		searchline_edit->show();
		cur_line++;
	}

	// Axes list

	Picviz::PVSource_p source = parent->current_tab->get_lib_view()->get_source_parent();
	QStringList axes_values;

	for (int i = 0; i < source->nraw->format->axes.size(); ++i) {
		axes_values << source->nraw->format->axes[i]["name"];
	}
	combo_axes_list = new QComboBox();
	QLabel *axes_label = new QLabel("Axis: ");

	combo_axes_list->addItems(axes_values);
	combo_axes_list->setCurrentIndex(axis_id);

	filter_widgets_layout->addWidget(axes_label, cur_line, 1);
	axes_label->show();

	filter_widgets_layout->addWidget(combo_axes_list, cur_line, 2);
	combo_axes_list->show();

	setLayout(main_layout);
	setWindowTitle(QString("Run filter ..."));
}

/******************************************************************************
 *
 * PVInspector::PVFilterSearchWidget::filter_cancel_action_Slot
 *
 *****************************************************************************/
void PVInspector::PVFilterSearchWidget::search_filter_cancel_action_Slot()
{
	reject();
}

/******************************************************************************
 *
 * PVInspector::PVFilterSearchWidget::filter_ok_action_Slot
 *
 *****************************************************************************/
void PVInspector::PVFilterSearchWidget::search_filter_ok_action_Slot()
{
	_axis_id = combo_axes_list->currentIndex();
	_regexp = searchline_edit->text();

	accept();
}
