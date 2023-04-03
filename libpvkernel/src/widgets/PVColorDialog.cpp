//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <pvkernel/core/PVPredefinedHSVColors.h>
#include <pvkernel/widgets/PVColorDialog.h>

#include <QMouseEvent>
#include <QAction>

// static PVCore::PVHSVColor g_predefined_colors[] = {HSV_COLOR_WHITE, HSV_COLOR_RED,
// HSV_COLOR_GREEN, HSV_COLOR_BLUE};

#define GRID_COL_SIZE 11
#define HSV_COLOR_PROPERTY "inendi_hsv_color_property"
#define HSV_COLOR_INDEX "inendi_hsv_color_index"

static void fill_label_with_color(QLabel* l, PVCore::PVHSVColor c)
{
	QColor qc = c.toQColor();

	QPixmap px(l->contentsRect().size());
	px.fill(qc);

	l->setPixmap(px);
}

static PVCore::PVHSVColor color_from_label(QLabel* label)
{
	return PVCore::PVHSVColor(label->property(HSV_COLOR_PROPERTY).toInt());
}

static size_t index_from_label(QLabel* label)
{
	return label->property(HSV_COLOR_INDEX).toInt();
}

static inline void select_label(QLabel* label)
{
	label->setLineWidth(3);
}

static inline void unselect_label(QLabel* label)
{
	label->setLineWidth(1);
}

static QLabel* label_from_sender(QObject* obj_sender)
{
	auto* const act = qobject_cast<QAction*>(obj_sender);
	if (!act) {
		return nullptr;
	}
	return qobject_cast<QLabel*>(act->parent());
}

namespace PVWidgets::__impl
{

class PVLabelEventFilter : public QObject
{
  public:
	explicit PVLabelEventFilter(PVWidgets::PVColorDialog* parent) : QObject(parent) {}

  protected:
	bool eventFilter(QObject* obj, QEvent* ev) override
	{
		assert(qobject_cast<QLabel*>(obj));
		auto* label = static_cast<QLabel*>(obj);
		switch (ev->type()) {
		case QEvent::MouseButtonPress:
			dlg_parent()->label_button_pressed(label, static_cast<QMouseEvent*>(ev));
			break;
		case QEvent::MouseButtonRelease:
			dlg_parent()->label_button_released(label, static_cast<QMouseEvent*>(ev));
			break;
		default:
			break;
		}

		return false;
	};

  private:
	inline PVWidgets::PVColorDialog* dlg_parent()
	{
		assert(qobject_cast<PVWidgets::PVColorDialog*>(parent()));
		return static_cast<PVWidgets::PVColorDialog*>(parent());
	}
};
} // namespace PVWidgets

PVWidgets::PVColorDialog::PVColorDialog(QWidget* parent)
    : PVColorDialog(PVCore::PVHSVColor(0), parent)
{
}

PVWidgets::PVColorDialog::PVColorDialog(PVCore::PVHSVColor const& c, QWidget* parent)
    : QDialog(parent)
{
	_label_event_filter = new __impl::PVLabelEventFilter(this);

	setupUi(this);

	// Init the predefined colors
	std::vector<PVCore::PVHSVColor> colors = PVCore::PVPredefinedHSVColors::get_predefined_colors();
	for (size_t i = 0; i < colors.size(); i++) {
		const PVCore::PVHSVColor c = colors[i];

		auto color_label = new QLabel();
		const QSize label_fixed_size(26, 26);
		color_label->setFrameShape(QFrame::Box);
		color_label->setFrameShadow(QFrame::Sunken);
		color_label->setMinimumSize(label_fixed_size);
		color_label->setMaximumSize(label_fixed_size);
		color_label->resize(label_fixed_size);
		color_label->setContextMenuPolicy(Qt::ActionsContextMenu);
		fill_label_with_color(color_label, c);

		// Add a property with the original color and index
		color_label->setProperty(HSV_COLOR_PROPERTY, QVariant(c.h()));
		color_label->setProperty(HSV_COLOR_INDEX, QVariant((int)i));

		// Install event filter...
		color_label->installEventFilter(_label_event_filter);

		// ...and other actions
		auto* act_save_color = new QAction(tr("Save current color"), color_label);
		connect(act_save_color, &QAction::triggered, this,
		        &PVColorDialog::set_predefined_color_from_action);
		auto* act_reset = new QAction(tr("Reset to white"), color_label);
		connect(act_reset, &QAction::triggered, this,
		        &PVColorDialog::reset_predefined_color_from_action);

		color_label->addAction(act_save_color);
		color_label->addAction(act_reset);

		_predefined_grid->addWidget(color_label, i / GRID_COL_SIZE, i % GRID_COL_SIZE,
		                            Qt::AlignLeft);
	}

	// Fill the last row with a spacer
	const size_t last_row = (colors.size() + (GRID_COL_SIZE - 1)) / GRID_COL_SIZE;
	const int last_empty_col = colors.size() % GRID_COL_SIZE;
	if (last_empty_col > 0) {
		_predefined_grid->addItem(new QSpacerItem(1, 0, QSizePolicy::Expanding), last_row,
		                          last_empty_col, GRID_COL_SIZE - last_empty_col);
	}

	connect(picker(), &PVColorPicker::color_changed_left, this,
	        &PVColorDialog::picker_color_changed);

	set_color(c);
}

void PVWidgets::PVColorDialog::picker_color_changed(int h)
{
	unselect_all_preselected_colors();
	// Change color of our label box
	show_color(PVCore::PVHSVColor(h));
	// Forward signal
	Q_EMIT color_changed(h);
}

void PVWidgets::PVColorDialog::show_color(PVCore::PVHSVColor const& c)
{
	fill_label_with_color(_box, c);
}

void PVWidgets::PVColorDialog::set_color(PVCore::PVHSVColor const& c)
{
	picker()->set_color(c);
	show_color(c);
}

void PVWidgets::PVColorDialog::unselect_all_preselected_colors()
{
	const size_t ncolors = PVCore::PVPredefinedHSVColors::get_predefined_colors_count();
	for (size_t i = 0; i < ncolors; i++) {
		auto* label = static_cast<QLabel*>(
		    _predefined_grid->itemAtPosition(i / GRID_COL_SIZE, i % GRID_COL_SIZE)->widget());
		unselect_label(label);
	}
}

void PVWidgets::PVColorDialog::label_button_pressed(QLabel* label, QMouseEvent* event)
{
	if (event->buttons() == Qt::LeftButton) {
		set_color(color_from_label(label));

		unselect_all_preselected_colors();
		select_label(label);
	}
}

void PVWidgets::PVColorDialog::label_button_released(QLabel* label, QMouseEvent* /*event*/)
{
	unselect_label(label);
}

void PVWidgets::PVColorDialog::set_predefined_color_from_action()
{
	QLabel* label = label_from_sender(sender());
	if (label) {
		set_predefined_color_from_label(label);
	}
}

void PVWidgets::PVColorDialog::reset_predefined_color_from_action()
{
	QLabel* label = label_from_sender(sender());
	if (label) {
		label->setProperty(HSV_COLOR_PROPERTY, HSV_COLOR_WHITE.h());
		fill_label_with_color(label, HSV_COLOR_WHITE);
	}
}

void PVWidgets::PVColorDialog::set_predefined_color_from_label(QLabel* label)
{
	const PVCore::PVHSVColor c = color();
	if (PVCore::PVPredefinedHSVColors::set_predefined_color(index_from_label(label), c)) {
		label->setProperty(HSV_COLOR_PROPERTY, QVariant(c.h()));
		fill_label_with_color(label, c);
	}
}
