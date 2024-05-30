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

#include "PVSaveDataTreeDialog.h"

#include <pvbase/general.h>

#include <QGridLayout>
#include <QCheckBox>

App::PVSaveDataTreeDialog::PVSaveDataTreeDialog(QString const& suffix,
                                                        QString const& filter,
                                                        QWidget* parent)
    : PVWidgets::PVFileDialog(parent)
{
	setAcceptMode(QFileDialog::AcceptSave);
	setDefaultSuffix(suffix);
	setWindowTitle(tr("Save project..."));
	setNameFilters(QStringList() << filter << ALL_FILES_FILTER);

	auto* main_layout = (QGridLayout*)layout();

	_save_everything_checkbox = new QCheckBox(tr("Include original files"), this);
	_save_everything_checkbox->setTristate(false);
	main_layout->addWidget(_save_everything_checkbox, 5, 1);
}

bool App::PVSaveDataTreeDialog::save_log_file() const
{
	return _save_everything_checkbox->isChecked();
}
