/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2020
 */

#include <pvguiqt/PVDisplayViewPythonConsole.h>
#include <pvkernel/widgets/PVFileDialog.h>

#include <inendi/PVRoot.h>

#include"pybind11/pybind11.h"
#include"pybind11/embed.h"
#include"pybind11/numpy.h"

#include <QTextEdit>
#include <QGridLayout>
#include <QPushButton>
#include <QLineEdit>
#include <QLabel>

#include <functional>

PVDisplays::PVDisplayViewPythonConsole::PVDisplayViewPythonConsole()
    : PVDisplayViewIf(PVDisplayIf::ShowInToolbar | PVDisplayIf::ShowInCentralDockWidget,
                      "Python console",
                      QIcon(":/python"),
                      Qt::NoDockWidgetArea)
{
}

static void run_python(const std::function<void()>& f, Inendi::PVPythonAppSingleton& python_interpreter, QTextEdit* console_output)
{
	auto start = std::chrono::system_clock::now();

	python_interpreter.python_output.clearStdout();
	try {
		f();
		console_output->setText(python_interpreter.python_output.stdoutString().c_str());
		console_output->setStyleSheet("QTextEdit { background-color : black; color : #00ccff; }");
	} catch (pybind11::error_already_set &eas) {
		console_output->setText(eas.what());
		console_output->setStyleSheet("QTextEdit { background-color : black; color : red; }");
	}

	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> diff = end - start;

	pvlogger::info() << "python script execution took : " << diff.count() << " ms" << std::endl << std::flush;
}

QWidget* PVDisplays::PVDisplayViewPythonConsole::create_widget(Inendi::PVView* view,
                                                               QWidget* parent,
                                                               Params const&) const
{
	// TODO : add drag n drop from axes/columns and sources

	Inendi::PVRoot& root = view->get_parent<Inendi::PVRoot>();
	Inendi::PVPythonAppSingleton& python_interpreter = root.python_interpreter();

	QWidget* console_widget = new QWidget(parent);

	QTextEdit* console_input = new QTextEdit(parent);
	console_input->setStyleSheet("QTextEdit { background-color : black; color : #ffcc00; }");

	QTextEdit* console_output = new QTextEdit(parent);
	console_output->setStyleSheet("QTextEdit { background-color : black; color : #00ccff; }");

	QPushButton* exec_script = new QPushButton("Exec sc&ript");
	QObject::connect(exec_script, &QPushButton::clicked, [=,&python_interpreter](){
		run_python([=](){
			pybind11::exec(console_input->toPlainText().toStdString());
		}, python_interpreter, console_output);
	});

	QLabel* exec_file_label = new QLabel("Python file:");
	QLineEdit* exec_file_line_edit = new QLineEdit();
	exec_file_line_edit->setReadOnly(true);
	QPushButton* exec_file_browse =  new QPushButton("&Browse...");
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
	QPushButton* exec_file_button = new QPushButton("Exec &file");
	QObject::connect(exec_file_button, &QPushButton::clicked, [=,&python_interpreter](){
		const QString& file_path = exec_file_line_edit->text();
		if (QFileInfo(file_path).exists()) {
			run_python([=, &python_interpreter](){
				pybind11::eval_file(file_path.toStdString());
			}, python_interpreter, console_output);
		}
	});

	QGridLayout* layout = new QGridLayout;

	layout->addWidget(exec_file_label, 0, 0);
	layout->addWidget(exec_file_line_edit, 0, 1);
	layout->addWidget(exec_file_browse, 0, 2);
	layout->addWidget(exec_file_button, 0, 3);

	layout->addWidget(console_input, 1, 0, 1, 3);
	layout->addWidget(exec_script, 1, 3);
	layout->addWidget(console_output, 2, 0, -1, -1);
	console_widget->setLayout(layout);

	return console_widget;
}
