#include <PVExpandSelDlg.h>
#include <QGridLayout>
#include <QDialogButtonBox>
#include <QLabel>
#include <QVBoxLayout>

#include <picviz/PVPlottingFilter.h>

PVInspector::PVExpandSelDlg::PVExpandSelDlg(Picviz::PVView_p view, QWidget* parent):
	QDialog(parent),
	_view(*view)
{
	setWindowTitle(tr("Expand selection..."));

	_axes_editor = new PVAxesIndexEditor(*view, this);
	PVCore::PVAxesIndexType axes;
	axes.push_back(0);
	_axes_editor->set_axes_index(axes);

	_combo_modes = new QComboBox();

	QVBoxLayout* main_layout = new QVBoxLayout();

	QGridLayout* grid_layout = new QGridLayout();
	grid_layout->addWidget(new QLabel(tr("Axes:"), this), 0, 0);
	grid_layout->addWidget(_axes_editor, 0, 1);
	grid_layout->addWidget(new QLabel(tr("Mode:"), this), 1, 0);
	grid_layout->addWidget(_combo_modes);

	main_layout->addLayout(grid_layout);

	QDialogButtonBox* btns = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
	main_layout->addWidget(btns);

	setLayout(main_layout);

	update_list_modes();

	connect(_axes_editor, SIGNAL(itemSelectionChanged()), this, SLOT(update_list_modes()));
	connect(btns, SIGNAL(accepted()), this, SLOT(accept()));
	connect(btns, SIGNAL(rejected()), this, SLOT(reject()));
}

void PVInspector::PVExpandSelDlg::update_list_modes()
{
	PVCore::PVAxesIndexType axes = _axes_editor->get_axes_index();

	QSet<QString> modes;
	PVCore::PVAxesIndexType::const_iterator it_axes;
	for (it_axes = axes.begin(); it_axes != axes.end(); it_axes++) {
		PVCol axis_id = *it_axes;
		QSet<QString> axis_modes = Picviz::PVPlottingFilter::list_modes(_view.get_axis_type(axis_id)).toSet();
		if (modes.size() == 0) {
			modes = axis_modes;
		}
		else {
			modes.intersect(axis_modes);
		}
	}

	QString cur_mode = "default";
	if (_combo_modes->count() > 0) {
		QString tmp = _combo_modes->currentText();
		if (modes.contains(tmp)) {
			cur_mode = tmp;
		}
	}

	_combo_modes->clear();
	QStringList list_modes = modes.toList();
	_combo_modes->addItems(list_modes);
	int idx = list_modes.indexOf(cur_mode);
	if (idx == -1) {
		idx = 0;
	}
	_combo_modes->setCurrentIndex(idx);
}

QString PVInspector::PVExpandSelDlg::get_mode()
{
	return _combo_modes->currentText();
}

PVCore::PVAxesIndexType PVInspector::PVExpandSelDlg::get_axes() const
{
	return _axes_editor->get_axes_index();
}
