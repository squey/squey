
#include <pvkernel/widgets/PVLayerNamingPatternDialog.h>

#include <QLineEdit>
#include <QComboBox>
#include <QDialogButtonBox>
#include <QLabel>
#include <QVBoxLayout>
#include <QHBoxLayout>

/******************************************************************************
 * PVWidgets::PVLayerNamingPatternDialog::PVLayerNamingPatternDialog
 *****************************************************************************/

PVWidgets::PVLayerNamingPatternDialog::PVLayerNamingPatternDialog(const QString& title,
                                                                  const QString& text,
                                                                  const QString& pattern,
                                                                  insert_mode m,
                                                                  QWidget* parent) :
	QDialog(parent)
{
	setWindowTitle(title);

	QVBoxLayout* vlayout = new QVBoxLayout();
	setLayout(vlayout);

	// the explaination text
	QLabel* label = new QLabel(text + "; substitution form:\n%l: current layer's name\n%a: axis' name\n%v: comma separated values");
	vlayout->addWidget(label);

	// the pattern edit
	_line_edit = new QLineEdit(pattern);
	vlayout->addWidget(_line_edit);

	// the placement choice
	QHBoxLayout* hlayout = new QHBoxLayout();
	vlayout->addLayout(hlayout);

	label = new QLabel("placement:");
	hlayout->addWidget(label);

	_combo_box = new QComboBox();
	_combo_box->addItem("On top of the layer stack");
	_combo_box->addItem("Above the current layer");
	_combo_box->addItem("Below the current layer");
	_combo_box->setCurrentIndex(m);
	hlayout->addWidget(_combo_box);

	// the button box
	QDialogButtonBox* button_box = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
	connect(button_box, SIGNAL(accepted()), this, SLOT(accept()));
	connect(button_box, SIGNAL(rejected()), this, SLOT(reject()));

	vlayout->addWidget(button_box);
}

/******************************************************************************
 * PVWidgets::PVLayerNamingPatternDialog::get_name_pattern
 *****************************************************************************/

QString PVWidgets::PVLayerNamingPatternDialog::get_name_pattern() const
{
	return _line_edit->text();
}

/******************************************************************************
 * PVWidgets::PVLayerNamingPatternDialog::get_insertion_mode
 *****************************************************************************/

PVWidgets::PVLayerNamingPatternDialog::insert_mode
PVWidgets::PVLayerNamingPatternDialog::get_insertion_mode() const
{
	return (insert_mode)_combo_box->currentIndex();
}
