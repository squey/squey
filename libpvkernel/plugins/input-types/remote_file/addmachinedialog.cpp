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

#include "include/addmachinedialog.h"

#include <QFormLayout>
#include <QVBoxLayout>
#include <QLineEdit>
#include <QLabel>
#include <QDialogButtonBox>
#include <QPushButton>

class AddMachineDialog::AddMachineDialogPrivate
{
  public:
	AddMachineDialogPrivate(AddMachineDialog* q) : machineName(nullptr), hostName(nullptr), buttons(nullptr), qq(q) {}
	void initWidget();
	void machineNameChanged();
	QLineEdit* machineName;
	QLineEdit* hostName;
	QDialogButtonBox* buttons;
	AddMachineDialog* qq;
};

void AddMachineDialog::AddMachineDialogPrivate::initWidget()
{
	QVBoxLayout* layout = new QVBoxLayout;

	QFormLayout* formLayout = new QFormLayout;
	layout->addLayout(formLayout);

	machineName = new QLineEdit;
	formLayout->addRow(tr("Machine name:"), machineName);
	connect(machineName, &QLineEdit::textChanged, qq, &AddMachineDialog::slotTextChanged);

	hostName = new QLineEdit;
	formLayout->addRow(tr("Hostname:"), hostName);
	connect(hostName, &QLineEdit::textChanged, qq, &AddMachineDialog::slotTextChanged);

	buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
	connect(buttons, &QDialogButtonBox::accepted, qq, &QDialog::accept);
	connect(buttons, &QDialogButtonBox::rejected, qq, &QDialog::reject);
	layout->addWidget(buttons);

	machineNameChanged();
	qq->setLayout(layout);
	qq->setWindowTitle(tr("Add new Machine"));
}

void AddMachineDialog::AddMachineDialogPrivate::machineNameChanged()
{
	buttons->button(QDialogButtonBox::Ok)
	    ->setEnabled(!hostName->text().isEmpty() && !machineName->text().isEmpty());
}

AddMachineDialog::AddMachineDialog(QWidget* parent)
    : QDialog(parent), d(new AddMachineDialogPrivate(this))
{
	d->initWidget();
	resize(250, 100);
}

AddMachineDialog::~AddMachineDialog()
{
	delete d;
}

QString AddMachineDialog::machineName() const
{
	return d->machineName->text();
}

QString AddMachineDialog::hostName() const
{
	return d->hostName->text();
}

void AddMachineDialog::slotTextChanged()
{
	d->machineNameChanged();
}
