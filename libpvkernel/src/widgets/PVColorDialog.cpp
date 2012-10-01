#include <pvkernel/widgets/PVColorDialog.h>


PVWidgets::PVColorDialog::PVColorDialog(QWidget* parent):
	QDialog(parent)
{
	init();
	set_color(0);
}

PVWidgets::PVColorDialog::PVColorDialog(PVCore::PVHSVColor const& c, QWidget* parent):
	QDialog(parent)
{
	init();
	set_color(c);
}

void PVWidgets::PVColorDialog::init()
{
	setupUi(this);
	connect(picker(), SIGNAL(color_changed_left(int)), this, SLOT(picker_color_changed(int)));
}

void PVWidgets::PVColorDialog::picker_color_changed(int h)
{
	// Change color of our label box
	show_color(PVCore::PVHSVColor(h));
	// Forward signal
	emit color_changed(h);
}

void PVWidgets::PVColorDialog::show_color(PVCore::PVHSVColor const c)
{
	QColor qc;
	c.toQColor(qc);

	QPixmap px(_box->contentsRect().size());
	px.fill(qc);

	_box->setPixmap(px);
}

void PVWidgets::PVColorDialog::set_color(PVCore::PVHSVColor const c)
{
	picker()->set_color(c);
	show_color(c);
}
