//
// MIT License
//
// © Squey, 2023
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

#include <pvguiqt/PVProgressBoxPython.h>
#include <squey/PVPythonSource.h>
#include <squey/PVScaled.h>

#include <boost/thread.hpp>
#include <boost/lexical_cast.hpp>
#include "pybind11/pybind11.h"

#include <QApplication>
#include <QLabel>
#include <QProgressBar>

PVCore::PVProgressBox::CancelState PVGuiQt::PVProgressBoxPython::progress(
	PVCore::PVProgressBox::process_t f,
	Squey::PVView* view,
	QString const& name,
	QString& exception_message,
	QWidget* parent)
{
	pybind11::gil_scoped_release gil{};

	PVProgressBoxPython pbox(name, parent);
    pbox.set_extended_status("(this progress box may hang for a while)");

	QObject::connect(&pbox, &PVCore::PVProgressBox::cancel_asked_sig, [&](){
		pbox.set_message("Canceling python script...");
		pbox.update();

		// Cancel python script execution
		auto threadId = boost::lexical_cast<std::string>(boost::this_thread::get_id());
		long long unsigned int threadNumber = 0;
		sscanf(threadId.c_str(), "%llx", &threadNumber);
		PyGILState_STATE gstate = PyGILState_Ensure();
		PyThreadState_SetAsyncExc(threadNumber, PyExc_InterruptedError);
		PyGILState_Release(gstate);
	});

	boost::thread th([&]() {
		try {
			// Execute python code in a cancelable thread
			pybind11::gil_scoped_acquire gil{};
			pybind11::module main = pybind11::module::import("__main__");
			main.attr(Squey::PVPythonSource::GUI_UPDATE_VAR) = pybind11::cast((uint32_t)Squey::PVPythonSource::GuiUpdateType::NONE);
			f(pbox);

			// Update the GUI in the main thread
            pbox.moveToThread(QCoreApplication::instance()->thread());
			auto update_type = main.attr(Squey::PVPythonSource::GUI_UPDATE_VAR).cast<uint32_t>();
			if (update_type & Squey::PVPythonSource::GuiUpdateType::SCALING) {
                QObject::connect(
					&pbox,
					&PVGuiQt::PVProgressBoxPython::emit_scaling_updated,
					&pbox,
					&PVGuiQt::PVProgressBoxPython::do_emit_scaling_updated,
					Qt::QueuedConnection
				);
                Q_EMIT pbox.emit_scaling_updated(view);
			}
			if (update_type & Squey::PVPythonSource::GuiUpdateType::LAYER) {
                QObject::connect(
					&pbox,
					&PVGuiQt::PVProgressBoxPython::emit_layer_updated,
					&pbox,
					&PVGuiQt::PVProgressBoxPython::do_emit_layer_updated,
					Qt::QueuedConnection
				);
                Q_EMIT pbox.emit_layer_updated(view);
			}
		} catch (const pybind11::error_already_set &eas) {
			Q_EMIT pbox.canceled_sig(); // dismiss progress box in GUI thread
			exception_message = eas.what();
			return;
		}
		Q_EMIT pbox.finished_sig(); // dismiss progress box in GUI thread
	});

	if (!th.timed_join(boost::posix_time::milliseconds(0))) {
		pbox.exec();
	}

	th.join();

	return pbox.get_cancel_state();
}

void PVGuiQt::PVProgressBoxPython::do_emit_scaling_updated(Squey::PVView* view)
{
    view->_axis_combination_updated.emit(false);
    view->get_parent<Squey::PVScaled>().update_scaling();
}


void PVGuiQt::PVProgressBoxPython::do_emit_layer_updated(Squey::PVView* view)
{
    view->_layer_stack_refreshed.emit();
    view->_update_current_min_max.emit();
}