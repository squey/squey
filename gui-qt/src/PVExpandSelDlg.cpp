/**
 * \file PVExpandSelDlg.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/core/hash_sharedptr.h>
#include <PVExpandSelDlg.h>
#include <QGridLayout>
#include <QDialogButtonBox>
#include <QLabel>
#include <QVBoxLayout>
#include <QPushButton>

#include <picviz/PVPlottingFilter.h>

PVInspector::PVExpandSelDlg::PVExpandSelDlg(Picviz::PVView_p view, QWidget* parent):
	QDialog(parent),
	_view(*view)
{
	setWindowTitle(tr("Expand selection..."));

	_axes_editor = new PVWidgets::PVAxesIndexEditor(*view, this);
	PVCore::PVAxesIndexType axes;
	axes.push_back(0);
	_axes_editor->set_axes_index(axes);

	_combo_modes = new QComboBox();

	QVBoxLayout* main_layout = new QVBoxLayout();

	QGridLayout* grid_layout = new QGridLayout();
	grid_layout->addWidget(new QLabel(tr("Axes :"), this), 0, 0);
	grid_layout->addWidget(_axes_editor, 0, 1);
	grid_layout->addWidget(new QLabel(tr("Mode :"), this), 1, 0);
	grid_layout->addWidget(_combo_modes);

	main_layout->addLayout(grid_layout);

	_btns = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
	main_layout->addWidget(_btns);

	setLayout(main_layout);

	update_list_modes();

	connect(_axes_editor, SIGNAL(itemSelectionChanged()), this, SLOT(update_list_modes()));
	connect(_btns, SIGNAL(accepted()), this, SLOT(accept()));
	connect(_btns, SIGNAL(rejected()), this, SLOT(reject()));
}

void PVInspector::PVExpandSelDlg::update_list_modes()
{
	PVCore::PVAxesIndexType axes = _axes_editor->get_axes_index();

	QSet<Picviz::PVPlottingFilter::p_type> modes;
	PVCore::PVAxesIndexType::const_iterator it_axes;
	for (it_axes = axes.begin(); it_axes != axes.end(); it_axes++) {
		PVCol axis_id = *it_axes;
		QSet<Picviz::PVPlottingFilter::p_type> axis_modes = Picviz::PVPlottingFilter::list_modes_lib(_view.get_original_axis_type(axis_id), true).toSet();
		if (modes.size() == 0) {
			modes = axis_modes;
		}
		else {
			modes.intersect(axis_modes);
		}
	}

	_combo_modes->clear();

	bool activate = modes.size() > 0;
	_combo_modes->setEnabled(activate);
	_btns->button(QDialogButtonBox::Ok)->setEnabled(activate);

	foreach (Picviz::PVPlottingFilter::p_type const& lib_f, modes) {
		_combo_modes->addItem(lib_f->get_human_name(), lib_f->registered_name());
	}

	_combo_modes->setCurrentIndex(0);
}

QString PVInspector::PVExpandSelDlg::get_mode()
{
	return Picviz::PVPlottingFilter::mode_from_registered_name(_combo_modes->itemData(_combo_modes->currentIndex()).toString());
}

PVCore::PVAxesIndexType PVInspector::PVExpandSelDlg::get_axes() const
{
	return _axes_editor->get_axes_index();
}
