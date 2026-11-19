/* * MIT License
 *
 * © Squey, 2026
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include <pvguiqt/PVAxesCombinationExportWidget.h>

#include <pvguiqt/PVAxesCombinationWidget.h>

#include <squey/PVSource.h>
#include <squey/PVView.h>

#include <QButtonGroup>
#include <QHBoxLayout>
#include <QPushButton>
#include <QRadioButton>
#include <QVBoxLayout>

PVGuiQt::PVAxesCombinationExportWidget::PVAxesCombinationExportWidget(Squey::PVView const& view,
                                                                      Choice choice,
                                                                      QString const& all_label,
                                                                      QWidget* parent)
    : QWidget(parent)
    , _view(view)
    , _custom_axes_combination(view.get_axes_combination())
    , _choice(choice)
{
	auto* all = new QRadioButton(all_label);
	auto* current = new QRadioButton(tr("Use current axes combination"));
	auto* custom = new QRadioButton(tr("Use custom axes combination"));
	auto* edit = new QPushButton(tr("Edit"));
	edit->setEnabled(choice == Choice::Custom);

	auto* group = new QButtonGroup(this);
	group->addButton(all, int(Choice::All));
	group->addButton(current, int(Choice::Current));
	group->addButton(custom, int(Choice::Custom));
	group->button(int(choice))->setChecked(true);

	// A window of its own, which goes with this widget, and with the combination it edits.
	auto* editor = new PVAxesCombinationWidget(_custom_axes_combination, nullptr, this);
	editor->setWindowFlags(Qt::Window);
	editor->setWindowModality(Qt::WindowModal);
	editor->hide();

	connect(group, &QButtonGroup::idToggled, this, [this, edit](int id, bool checked) {
		if (checked) {
			_choice = Choice(id);
			edit->setEnabled(_choice == Choice::Custom);
			Q_EMIT changed();
		}
	});
	connect(edit, &QPushButton::clicked, editor, &QWidget::show);
	connect(editor, &PVAxesCombinationWidget::closed, this, [this] {
		if (_choice == Choice::Custom) {
			Q_EMIT changed();
		}
	});

	auto* custom_layout = new QHBoxLayout;
	custom_layout->addWidget(custom);
	custom_layout->addWidget(edit);

	auto* layout = new QVBoxLayout(this);
	layout->setContentsMargins(0, 0, 0, 0);
	layout->addWidget(all);
	layout->addWidget(current);
	layout->addLayout(custom_layout);
}

PVCore::PVColumnIndexes PVGuiQt::PVAxesCombinationExportWidget::column_indexes() const
{
	switch (_choice) {
	case Choice::All: {
		PVCore::PVColumnIndexes columns;
		for (PVCol column(0); column < _view.get_parent<Squey::PVSource>().get_nraw_column_count();
		     column++) {
			columns.push_back(column);
		}
		return columns;
	}
	case Choice::Current:
		return _view.get_axes_combination().get_combination();
	case Choice::Custom:
		return _custom_axes_combination.get_combination();
	}
	return {};
}

QStringList PVGuiQt::PVAxesCombinationExportWidget::names() const
{
	switch (_choice) {
	case Choice::All:
		return _view.get_axes_combination().get_nraw_names();
	case Choice::Current:
		return _view.get_axes_combination().get_combined_names();
	case Choice::Custom:
		return _custom_axes_combination.get_combined_names();
	}
	return {};
}
