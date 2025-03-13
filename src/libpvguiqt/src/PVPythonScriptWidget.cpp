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

#include <pvguiqt/PVPythonScriptWidget.h>

#include <pvkernel/widgets/PVFileDialog.h>
#include <pvguiqt/PVAboutBoxDialog.h>
#include <pvguiqt/PVDisplayViewPythonConsole.h>
#include <pvguiqt/PVPythonCodeEditor.h>

#include <QDesktopServices>
#include <QGroupBox>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QSpacerItem>
#include <QLineEdit>
#include <QTextEdit>
#include <QPushButton>
#include <QRadioButton>
#include <QButtonGroup>
#include <QStandardPaths>

PVGuiQt::PVPythonScriptWidget::PVPythonScriptWidget(QWidget* parent /*= nullptr*/)
    : QGroupBox(tr("Execute Python script after import"), parent)
{
    setCheckable(true);
    setChecked(false);

	auto* layout = new QVBoxLayout(this);

    // Script path
	auto* python_script_path_layout = new QHBoxLayout();
	auto* python_script_path_container_widget = new QWidget;
	python_script_path_container_widget->setLayout(python_script_path_layout);
	_python_script_path_radio = new QRadioButton();
	_python_script_path_radio->setAutoExclusive(true);
	auto* exec_python_file_label = new QLabel("Python file:");
	_exec_python_file_line_edit = new QLineEdit();
	_exec_python_file_line_edit->setReadOnly(true);
	auto* exec_python_file_browse =  new QPushButton("&Browse...");
	QObject::connect(exec_python_file_browse, &QPushButton::clicked, [=,this]() {
		QString file_path = PVWidgets::PVFileDialog::getOpenFileName(
			this,
			"Browse your python file",
			QStandardPaths::writableLocation(QStandardPaths::HomeLocation),
			QString("Python file (*.py)"));
		if (not file_path.isEmpty()) {
			_exec_python_file_line_edit->setText(file_path);
            _python_script_path_radio->setChecked(true);
            notify_python_script_updated();
		}
	});
	python_script_path_layout->addWidget(_exec_python_file_line_edit);
	python_script_path_layout->addWidget(exec_python_file_browse);
    auto* python_script_path_radio_layout = new QHBoxLayout();
    python_script_path_container_widget->setLayout(python_script_path_layout);
    python_script_path_radio_layout->addWidget(_python_script_path_radio);
    python_script_path_radio_layout->addWidget(exec_python_file_label);
    python_script_path_radio_layout->addWidget(python_script_path_container_widget);

    // Script content
	auto* python_script_content_layout = new QHBoxLayout();
	auto* python_script_content_container_widget = new QWidget;
	_python_script_content_radio = new QRadioButton();
	_python_script_content_radio->setAutoExclusive(true);
	auto* exec_python_content_label = new QLabel("Python script:");
	_python_script_content_text = new PVGuiQt::PVPythonCodeEditor(parent);;

    connect(_python_script_content_text, &QTextEdit::textChanged, this, [this](){
        notify_python_script_updated();
    });
	python_script_content_layout->addWidget(_python_script_content_text);
    auto* python_script_content_radio_layout = new QHBoxLayout();
    python_script_content_container_widget->setLayout(python_script_content_layout);
    python_script_content_radio_layout->addWidget(_python_script_content_radio);
    python_script_content_radio_layout->addWidget(exec_python_content_label);
    python_script_content_radio_layout->addWidget(python_script_content_container_widget);

	auto* python_script_radio_group = new QButtonGroup(this);
	python_script_radio_group->addButton(_python_script_path_radio);
	python_script_radio_group->addButton(_python_script_content_radio);

	connect(_python_script_content_radio, &QRadioButton::toggled, this, [=,this]() {
		bool checked = _python_script_content_radio->isChecked();
		python_script_path_container_widget->setEnabled(not checked);
        if (isChecked()) {
		    python_script_content_container_widget->setEnabled(checked);
        }
        notify_python_script_updated();
	});
	connect(_python_script_path_radio, &QRadioButton::toggled, this, [=,this]() {
		bool checked = _python_script_path_radio->isChecked();
        if (isChecked()) {
            python_script_path_container_widget->setEnabled(checked);
        }
		python_script_content_container_widget->setEnabled(not checked);
        notify_python_script_updated();
	});

    connect(this, &QGroupBox::toggled, this, [this](){
        notify_python_script_updated();
    });

    // Help
    auto* help_button = new QPushButton("&Help");
    connect(help_button, &QPushButton::clicked, []() {
        QDesktopServices::openUrl(QUrl(QString(DOC_URL) + "/content/python_scripting/content.html"));
    });

	layout->addLayout(python_script_path_radio_layout);
	layout->addLayout(python_script_content_radio_layout);
    layout->addWidget(help_button);
}


void PVGuiQt::PVPythonScriptWidget::set_python_script(const QString& python_script, bool is_path, bool disabled)
{
    _python_script_path_radio->setChecked(is_path); 
    _python_script_content_radio->setChecked(not is_path); 
    if (is_path) {
        _exec_python_file_line_edit->setText(python_script);
    }
    else {
        _python_script_content_text->setText(python_script);
    }
    setChecked(not disabled);
}

QString PVGuiQt::PVPythonScriptWidget::get_python_script(bool& is_path, bool& disabled) const
{
    is_path = _python_script_path_radio->isChecked();
    disabled = not isChecked();
    if (is_path) {
        return _exec_python_file_line_edit->text();
    }
    else {
        return _python_script_content_text->toPlainText();
    }
}

void PVGuiQt::PVPythonScriptWidget::notify_python_script_updated()
{
    bool is_path, disabled;
    const QString& python_script = get_python_script(is_path, disabled);

    Q_EMIT python_script_updated(python_script, is_path, disabled);
}
