/**
 * \file PVAxisIndexCheckBoxEditor.cpp
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVAxisIndexType.h>

#include <picviz/PVView.h>

#include <picviz/widgets/editors/PVAxisIndexCheckBoxEditor.h>

#include <QHBoxLayout>



/******************************************************************************
 *
 * PVCore::PVAxisIndexCheckBoxEditor::PVAxisIndexCheckBoxEditor
 *
 *****************************************************************************/
PVWidgets::PVAxisIndexCheckBoxEditor::PVAxisIndexCheckBoxEditor(Picviz::PVView const& view, QWidget *parent):
	QWidget(parent),
	_view(view)
{
	// _checked = true;	// Default is checked
	// _current_index = 0;

	QHBoxLayout *layout = new QHBoxLayout;
	checkbox = new QCheckBox;
	checkbox->setCheckState(Qt::Checked);
	layout->addWidget(checkbox);
	// combobox = new QComboBox;
	// layout->addWidget(combobox);

	setLayout(layout);

	// connect(checkbox, SIGNAL(stateChanged(int)), this, SLOT(checkStateChanged_Slot(int)));

	setFocusPolicy(Qt::StrongFocus);
}

/******************************************************************************
 *
 * PVWidgets::PVAxisIndexCheckBoxEditor::~PVAxisIndexCheckBoxEditor
 *
 *****************************************************************************/
PVWidgets::PVAxisIndexCheckBoxEditor::~PVAxisIndexCheckBoxEditor()
{
}

/******************************************************************************
 *
 * PVWidgets::PVAxisIndexCheckBoxEditor::set_axis_index
 *
 *****************************************************************************/
void PVWidgets::PVAxisIndexCheckBoxEditor::set_axis_index(PVCore::PVAxisIndexCheckBoxType /*axis_index*/)
{
	PVLOG_INFO("WE SET THE INDEX OF OUR CHECKBOX FROM THE EDITOR!\n");

	// combobox->clear();
	// combobox->addItems(_view.get_axes_names_list());
	// combobox->setCurrentIndex(axis_index.get_original_index());
}

PVCore::PVAxisIndexCheckBoxType PVWidgets::PVAxisIndexCheckBoxEditor::get_axis_index() const
{
	// 1 should be replace by the check to know if it is checked
	// return PVCore::PVAxisIndexCheckBoxType(currentIndex(), is_checked());
	PVLOG_INFO("WE GET THE INDEX OF CHECK BOX FROM THE EDITOR\n");

	return PVCore::PVAxisIndexCheckBoxType(combobox->currentIndex(), 0);
}

