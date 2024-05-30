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

#include <iostream>
#include <pvguiqt/PVCredentialDialog.h>

#include <QVBoxLayout>
#include <QGroupBox>
#include <QFormLayout>
#include <QDialogButtonBox>

namespace PVGuiQt
{

CredentialDialog::CredentialDialog(QWidget* parent) : QDialog(parent)
{
	hide();

	auto layout = new QVBoxLayout();

	auto gridGroupBox = new QGroupBox("Provide your login and password", this);

	auto form = new QFormLayout(this);
	form->addRow("Login :", &_login);
	form->addRow("Password :", &_passwd);
	_passwd.setEchoMode(QLineEdit::Password);

	auto buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);

	connect(buttonBox, &QDialogButtonBox::accepted, this, &QDialog::accept);
	connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);

	gridGroupBox->setLayout(form);

	layout->addWidget(gridGroupBox);
	layout->addWidget(buttonBox);

	setLayout(layout);

	setWindowTitle("Credential");
}

} // namespace PVGuiQt
