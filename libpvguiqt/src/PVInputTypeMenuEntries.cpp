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

#include <pvguiqt/PVInputTypeMenuEntries.h>
#include <pvkernel/rush/PVInputType.h>
#include <pvkernel/core/PVClassLibrary.h>

#include <QPushButton>

/**
 * a map function to iterate over input plugins according to their sorted
 * ::internal_name().
 */
template <typename F>
static void map_input_by_sorted_internal_name(const F& f)
{
	LIB_CLASS(PVRush::PVInputType)& input_types = LIB_CLASS(PVRush::PVInputType)::get();
	LIB_CLASS(PVRush::PVInputType)::list_classes const& lf = input_types.get_list();

	using ele_t = std::pair<QString, QString>;
	std::list<ele_t> pairs;

	for (const auto& it : lf) {
		pairs.push_back(std::make_pair(it.value()->internal_name(), it.value()->registered_name()));
	}

	pairs.sort([](const ele_t& a, const ele_t& b) { return a.first.compare(b.first) < 0; });

	for (const auto& it : pairs) {
		f(it.second, lf.at(it.second));
	}
}

void PVGuiQt::PVInputTypeMenuEntries::add_inputs_to_menu(QMenu* menu,
                                                         QObject* parent,
                                                         const char* slot)
{
	map_input_by_sorted_internal_name([&](const QString& key, const PVRush::PVInputType_p& in) {
		QAction* action = new QAction(in->menu_input_name(), parent);
		action->setData(QVariant(key));
		action->setShortcut(in->menu_shortcut());
		QObject::connect(action, SIGNAL(triggered()), parent, slot);
		menu->addAction(action);
	});
}

void PVGuiQt::PVInputTypeMenuEntries::add_inputs_to_layout(QBoxLayout* layout,
                                                           QObject* parent,
                                                           const char* slot)
{
	map_input_by_sorted_internal_name([&](const QString& key, const PVRush::PVInputType_p& in) {
		QAction* action = new QAction(in->menu_input_name(), parent);
		action->setData(QVariant(key));
		action->setShortcut(in->menu_shortcut());
		QObject::connect(action, SIGNAL(triggered()), parent, slot);

		QPushButton* button = new QPushButton(in->menu_input_name());
		button->setIcon(in->icon());
		button->setCursor(in->cursor());

		QObject::connect(button, &QAbstractButton::clicked, action, &QAction::trigger);

		layout->addWidget(button);
	});
}

PVRush::PVInputType_p PVGuiQt::PVInputTypeMenuEntries::input_type_from_action(QAction* action)
{
	QString const& itype = action->data().toString();
	PVRush::PVInputType_p in_t = LIB_CLASS(PVRush::PVInputType)::get().get_class_by_name(itype);
	return in_t;
}
