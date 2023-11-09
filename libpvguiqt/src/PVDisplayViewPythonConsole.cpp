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

#include <pvguiqt/PVDisplayViewPythonConsole.h>
#include <pvguiqt/PVAboutBoxDialog.h>
#include <pvguiqt/PVPythonCodeEditor.h>
#include <pvguiqt/PVProgressBoxPython.h>
#include <pvkernel/core/PVProgressBox.h>
#include <pvkernel/widgets/PVFileDialog.h>

#include <squey/PVRoot.h>
#include <squey/PVPythonSource.h>

#include"pybind11/pybind11.h"
#include"pybind11/embed.h"
#include"pybind11/numpy.h"

#include <QApplication>
#include <QDesktopServices>
#include <QTextEdit>
#include <QGridLayout>
#include <QPushButton>
#include <QLineEdit>
#include <QLabel>


#include <functional>

#include <boost/thread.hpp>

PVDisplays::PVDisplayViewPythonConsole::PVDisplayViewPythonConsole()
    : PVDisplayViewIf(PVDisplayIf::ShowInToolbar | PVDisplayIf::ShowInCentralDockWidget,
                      "Python console",
                      QIcon(":/python"),
                      Qt::NoDockWidgetArea)
{
}

static void run_python(const std::function<void()>& f, Squey::PVPythonInterpreter& python_interpreter, Squey::PVView* view, QTextEdit* console_output, QWidget* /*parent*/)
{
	auto start = std::chrono::system_clock::now();

	QString exception_msg;
	PVCore::PVProgressBox::CancelState cancel_state = PVGuiQt::PVProgressBoxPython::progress([&](PVCore::PVProgressBox& pbox) {
		pbox.set_enable_cancel(true);
		try {
			f();
		} catch (const pybind11::error_already_set &eas) {
			if (eas.matches(PyExc_InterruptedError)) {
				pbox.set_canceled();
			}
			throw eas; // rethrow exception to handle progress box dismiss
		}
	}, view, QString("Executing python script..."), exception_msg, nullptr);

	if (cancel_state == PVCore::PVProgressBox::CancelState::CONTINUE) {
		if (exception_msg.isEmpty()) {
			console_output->setText(python_interpreter.python_output.stdoutString().c_str());
			python_interpreter.python_output.clearStdout();
			console_output->setStyleSheet("QTextEdit { background-color : black; color : #00ccff; }");
		}
		else {
			console_output->setText(exception_msg);
			console_output->setStyleSheet("QTextEdit { background-color : black; color : red; }");
		}
	}
	else {
		console_output->clear();
	}

	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> diff = end - start;

	pvlogger::info() << "python script execution took : " << diff.count() << " ms" << std::endl << std::flush;
}

QWidget* PVDisplays::PVDisplayViewPythonConsole::create_widget(Squey::PVView* view,
                                                               QWidget* parent,
                                                               Params const&) const
{
	auto& root = view->get_parent<Squey::PVRoot>();
	Squey::PVPythonInterpreter& python_interpreter = Squey::PVPythonInterpreter::get(root);

	auto* console_widget = new QWidget(parent);

	auto* console_input = new PVGuiQt::PVPythonCodeEditor(PVGuiQt::PVPythonCodeEditor::EThemeType::DARK, parent);

	auto* console_output = new QTextEdit(parent);
	console_output->setStyleSheet("QTextEdit { background-color : black; color : #00ccff; }");

	auto* exec_script = new QPushButton("Exec sc&ript");
	QObject::connect(exec_script, &QPushButton::clicked, [=,&python_interpreter](){
		run_python([=,&python_interpreter](){
			python_interpreter.execute_script(console_input->toPlainText().toStdString(), false);
		}, python_interpreter, view, console_output, parent);
	});

	auto* exec_file_label = new QLabel("Python file:");
	auto* exec_file_line_edit = new QLineEdit();
	exec_file_line_edit->setReadOnly(true);
	auto* exec_file_browse =  new QPushButton("&Browse...");
	QObject::connect(exec_file_browse, &QPushButton::clicked, [=]() {
		QString file_path = PVWidgets::PVFileDialog::getOpenFileName(
			console_widget,
			"Browse your python file",
			"",
			QString("Python file (*.py)"));
		if (not file_path.isEmpty()) {
			exec_file_line_edit->setText(file_path);
		}
	});
	auto* exec_file_button = new QPushButton("Exec &file");
	QObject::connect(exec_file_button, &QPushButton::clicked, [=,&python_interpreter](){
		const QString& file_path = exec_file_line_edit->text();
		if (QFileInfo(file_path).exists()) {
			run_python([=,&python_interpreter](){
				python_interpreter.execute_script(file_path.toStdString(), true);
			}, python_interpreter, view, console_output, parent);
		}
	});

    // Help
    auto* help_button = new QPushButton("&Help");
    QObject::connect(help_button, &QPushButton::clicked, []() {
        QDesktopServices::openUrl(QUrl(QString(DOC_URL) + "/content/python_scripting/content.html"));
    });

	auto* layout = new QGridLayout;

	layout->addWidget(exec_file_label, 0, 0);
	layout->addWidget(exec_file_line_edit, 0, 1);
	layout->addWidget(exec_file_browse, 0, 2);
	layout->addWidget(exec_file_button, 0, 3);

	layout->addWidget(console_input, 1, 0, 1, 3);
	layout->addWidget(exec_script, 1, 3);
	layout->addWidget(console_output, 2, 0, -1, -1);
	layout->addWidget(help_button, 3, 0, -1, -1);

	console_widget->setLayout(layout);

	return console_widget;
}
