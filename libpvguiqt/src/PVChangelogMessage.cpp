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

#include <pvguiqt/PVChangelogMessage.h>
#include <pvguiqt/PVAboutBoxDialog.h>

#include <pvkernel/core/PVConfig.h>

#include <QDir>
#include <QFile>
#include <QTextStream>

#include INENDI_VERSION_FILE_PATH

PVGuiQt::PVChangelogMessage::PVChangelogMessage(QWidget* parent /* = nullptr*/)
{
	QString current_version = QString(INENDI_CURRENT_VERSION_STR);
	QString previous_version;

	QFile version_file(QString::fromStdString(PVCore::PVConfig::user_dir()) + QDir::separator() +
	                   "version.txt");
	if (version_file.open(QFile::ReadOnly | QFile::Text)) {
		QTextStream in(&version_file);
		previous_version = in.readLine();
		version_file.close();
	}

	if (current_version != previous_version) {
		PVGuiQt::PVAboutBoxDialog aboutbox(PVGuiQt::PVAboutBoxDialog::Tab::CHANGELOG, parent);
		aboutbox.exec();
	}

	if (version_file.open(QIODevice::WriteOnly)) {
		QTextStream out(&version_file);
		out << current_version;
	}
}
