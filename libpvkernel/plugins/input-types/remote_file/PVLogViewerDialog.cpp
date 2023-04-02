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

#include "include/PVLogViewerDialog.h"

#include <QLabel>
#include <QMenu>
#include <QMenuBar>
#include <QDialogButtonBox>
#include <QWidget>
#include <QVBoxLayout>
#include <QSpacerItem>

PVLogViewerDialog::PVLogViewerDialog(QStringList const& formats, QWidget* parent) : QDialog(parent)
{
	QMenuBar* rl_menuBar = new QMenuBar(nullptr);
	QMenu* rl_fileMenu = rl_menuBar->addMenu(tr("Machine"));

	QDialogButtonBox* buttonBox =
	    new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);

	QVBoxLayout* rl_layout = new QVBoxLayout;

	pv_RemoteLog = new LogViewerWidget(this);
	rl_fileMenu->addAction(pv_RemoteLog->addMachineAction());
	rl_fileMenu->addAction(pv_RemoteLog->removeMachineAction());

	rl_layout->setMenuBar(rl_menuBar);
	rl_menuBar->show();

	rl_layout->addWidget(pv_RemoteLog);

	QHBoxLayout* formatLayout = new QHBoxLayout();
	formatLayout->addWidget(new QLabel(tr("Format:")));
	_combo_format = new QComboBox();
	_combo_format->addItems(formats);
	formatLayout->addWidget(_combo_format);
	formatLayout->addItem(new QSpacerItem(1, 1, QSizePolicy::Expanding, QSizePolicy::Minimum));

	rl_layout->addLayout(formatLayout);
	rl_layout->addWidget(buttonBox);

	connect(buttonBox, &QDialogButtonBox::accepted, this, &PVLogViewerDialog::slotDownloadFiles);
	connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);

	setLayout(rl_layout);
	setWindowTitle(tr("Import remote file"));
}

PVLogViewerDialog::~PVLogViewerDialog()
{
	pv_RemoteLog->deleteLater();
}

void PVLogViewerDialog::slotDownloadFiles()
{
	QHash<QString, QUrl> tmp_files;
	if (!pv_RemoteLog->downloadSelectedFiles(tmp_files)) {
		return;
	}

	_dl_files = tmp_files;
	_format = _combo_format->currentText();
	accept();
}

QString PVLogViewerDialog::getSelFormat()
{
	return _format;
}
