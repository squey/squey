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
