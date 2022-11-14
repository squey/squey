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

#include "include/fileconnectiondialog.h"
#include "include/logviewerwidget.h"
#include "include/logviewerwidget_p.h"

#include <QVBoxLayout>
#include <QDialogButtonBox>
#include <QComboBox>
#include <QGridLayout>
#include <QLabel>
#include <QSpinBox>
#include <QLineEdit>
#include <QPushButton>
#include <QCheckBox>
#include <QFormLayout>

FileNameSelectorWidget::FileNameSelectorWidget(QWidget* parent) : QWidget(parent)
{
	QHBoxLayout* layout = new QHBoxLayout;
	m_path = new QLineEdit;
	layout->addWidget(m_path);
	layout->setContentsMargins(0, 0, 0, 0);

	m_selectPath = new QPushButton;
	m_selectPath->setText(QLatin1String("..."));
	m_selectPath->setFixedWidth(30);
	connect(m_selectPath, &QAbstractButton::clicked, this,
	        &FileNameSelectorWidget::slotPathChanged);

	layout->addWidget(m_selectPath);

	setLayout(layout);
}

FileNameSelectorWidget::~FileNameSelectorWidget()
{
}

QString FileNameSelectorWidget::text() const
{
	return m_path->text();
}

void FileNameSelectorWidget::setText(const QString& text)
{
	m_path->setText(text);
}

void FileNameSelectorWidget::slotPathChanged()
{
	const QString fileName = _file_dlg.getOpenFileName(this, tr("SSH key file"));
	if (!fileName.isEmpty())
		m_path->setText(fileName);
}

class FileConnectionDialog::FileConnectionDialogPrivate
{
  public:
	FileConnectionDialogPrivate(FileConnectionDialog* q)
	    : protocols(0)
	    , port(0)
	    , hostname(0)
	    , remotefile(0)
	    , sshkey(0)
	    , certificate(0)
	    , ignoreSslError(0)
	    , password(0)
	    , login(0)
	    , buttons(0)
	    , formLayout(0)
	    , qq(q)
	{
	}

	void initWidget();
	void remoteFileChanged(const QString&);
	void protocolChanged(int index);
	void setFieldEnabled(QWidget* field, bool enabled);

	QComboBox* protocols;
	QSpinBox* port;
	QLabel* hostname;
	QLineEdit* remotefile;
	FileNameSelectorWidget* sshkey;
	FileNameSelectorWidget* certificate;
	QCheckBox* ignoreSslError;
	QLineEdit* password;
	QLineEdit* login;
	QDialogButtonBox* buttons;
	QFormLayout* formLayout;
	FileConnectionDialog* qq;
};

void FileConnectionDialog::FileConnectionDialogPrivate::initWidget()
{
	QVBoxLayout* layout = new QVBoxLayout;

	formLayout = new QFormLayout;
	layout->addLayout(formLayout);

	hostname = new QLabel;
	formLayout->addRow(tr("Hostname:"), hostname);

	protocols = new QComboBox;
	formLayout->addRow(tr("Protocol:"), protocols);
	protocols->addItems(LogViewerPrivate::defaultStringI18nProtocol);
	connect(protocols, SIGNAL(currentIndexChanged(int)), qq, SLOT(slotProtocolChanged(int)));

	remotefile = new QLineEdit;
	formLayout->addRow(tr("Remote File:"), remotefile);
	connect(remotefile, &QLineEdit::textChanged, qq, &FileConnectionDialog::slotTextChanged);

	port = new QSpinBox;
	port->setRange(0, 65535);
	formLayout->addRow(tr("Port:"), port);

	login = new QLineEdit;
	formLayout->addRow(tr("Login:"), login);

	password = new QLineEdit;
	password->setEchoMode(QLineEdit::Password);
	formLayout->addRow(tr("Password:"), password);

	sshkey = new FileNameSelectorWidget;
	formLayout->addRow(tr("SSH key File:"), sshkey);

	certificate = new FileNameSelectorWidget;
	formLayout->addRow(tr("Certificate File:"), certificate);

	ignoreSslError = new QCheckBox(tr("Ignore Ssl error"));
	layout->addWidget(ignoreSslError);

	buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
	connect(buttons, &QDialogButtonBox::accepted, qq, &QDialog::accept);
	connect(buttons, &QDialogButtonBox::rejected, qq, &QDialog::reject);
	layout->addWidget(buttons);
	remoteFileChanged(QString());
	protocolChanged(0);
	qq->setLayout(layout);
	qq->setWindowTitle(tr("Settings"));
}

void FileConnectionDialog::FileConnectionDialogPrivate::setFieldEnabled(QWidget* field,
                                                                        bool enabled)
{
	field->setEnabled(enabled);
	formLayout->labelForField(field)->setEnabled(enabled);
}

void FileConnectionDialog::FileConnectionDialogPrivate::protocolChanged(int index)
{
	const Protocol protocol = static_cast<Protocol>(index);
	switch (protocol) {
	case Local:
		port->setValue(22);
		setFieldEnabled(password, false);
		setFieldEnabled(sshkey, false);
		setFieldEnabled(login, false);
		setFieldEnabled(port, false);
		setFieldEnabled(certificate, false);
		ignoreSslError->setEnabled(false);
		break;
	case Http:
		port->setValue(80);
		setFieldEnabled(password, false);
		setFieldEnabled(sshkey, false);
		setFieldEnabled(login, false);
		setFieldEnabled(port, true);
		setFieldEnabled(certificate, false);
		ignoreSslError->setEnabled(false);
		break;
	case Https:
		port->setValue(443);
		setFieldEnabled(password, true);
		setFieldEnabled(sshkey, false);
		setFieldEnabled(login, true);
		setFieldEnabled(port, true);
		setFieldEnabled(certificate, true);
		ignoreSslError->setEnabled(true);
		break;
	case Ftp:
		port->setValue(21);
		setFieldEnabled(password, true);
		setFieldEnabled(sshkey, false);
		setFieldEnabled(login, true);
		setFieldEnabled(port, true);
		setFieldEnabled(certificate, false);
		ignoreSslError->setEnabled(false);
		break;
	case Ftps:
		port->setValue(990);
		setFieldEnabled(password, true);
		setFieldEnabled(sshkey, true);
		setFieldEnabled(login, true);
		setFieldEnabled(port, true);
		setFieldEnabled(certificate, true);
		ignoreSslError->setEnabled(true);
		break;
	case Scp:
		port->setValue(22);
		setFieldEnabled(password, true);
		setFieldEnabled(sshkey, true);
		setFieldEnabled(login, true);
		setFieldEnabled(port, true);
		setFieldEnabled(certificate, true);
		ignoreSslError->setEnabled(true);
		break;
	case SFtp:
		port->setValue(22);
		setFieldEnabled(password, true);
		setFieldEnabled(sshkey, true);
		setFieldEnabled(login, true);
		setFieldEnabled(port, true);
		setFieldEnabled(certificate, true);
		ignoreSslError->setEnabled(true);
		break;
	}
}

void FileConnectionDialog::FileConnectionDialogPrivate::remoteFileChanged(const QString& text)
{
	buttons->button(QDialogButtonBox::Ok)->setEnabled(!text.isEmpty());
}

FileConnectionDialog::FileConnectionDialog(QWidget* parent)
    : QDialog(parent), d(new FileConnectionDialogPrivate(this))
{
	d->initWidget();
	resize(400, 200);
}

FileConnectionDialog::~FileConnectionDialog()
{
	delete d;
}

RegisteredFile FileConnectionDialog::registeredFileSettings() const
{
	RegisteredFile registered;
	registered.remoteFile = d->remotefile->text();
	const Protocol protocol = static_cast<Protocol>(d->protocols->currentIndex());
	registered.settings.protocol = protocol;

	switch (protocol) {
	case Local:
		break;
	case Http: {
		registered.settings.port = d->port->value();
		registered.settings.login = d->login->text();
	} break;
	case Https: {
		registered.settings.port = d->port->value();
		registered.settings.sshKeyFile = d->sshkey->text();
		registered.settings.password = d->password->text();
		registered.settings.login = d->login->text();
		registered.settings.certificatFile = d->certificate->text();
		registered.settings.ignoreSslError = d->ignoreSslError->isChecked();
	} break;
	case Ftp: {
		registered.settings.port = d->port->value();
		registered.settings.login = d->login->text();
	} break;
	case Ftps: {
		registered.settings.port = d->port->value();
		registered.settings.sshKeyFile = d->sshkey->text();
		registered.settings.password = d->password->text();
		registered.settings.login = d->login->text();
		registered.settings.certificatFile = d->certificate->text();
		registered.settings.ignoreSslError = d->ignoreSslError->isChecked();
	} break;
	case Scp: {
		registered.settings.port = d->port->value();
		registered.settings.sshKeyFile = d->sshkey->text();
		registered.settings.password = d->password->text();
		registered.settings.login = d->login->text();
		registered.settings.certificatFile = d->certificate->text();
		registered.settings.ignoreSslError = d->ignoreSslError->isChecked();
	} break;
	case SFtp: {
		registered.settings.port = d->port->value();
		registered.settings.sshKeyFile = d->sshkey->text();
		registered.settings.password = d->password->text();
		registered.settings.login = d->login->text();
		registered.settings.certificatFile = d->certificate->text();
		registered.settings.ignoreSslError = d->ignoreSslError->isChecked();
	} break;
	default:
		break;
	}
	return registered;
}

void FileConnectionDialog::initialize(const RegisteredFile& registered, const QString& hostName)
{
	d->protocols->setCurrentIndex(registered.settings.protocol);
	d->hostname->setText(hostName);
	d->port->setValue(registered.settings.port);
	d->remotefile->setText(registered.remoteFile);
	d->sshkey->setText(registered.settings.sshKeyFile);
	d->password->setText(registered.settings.password);
	d->login->setText(registered.settings.login);
	d->certificate->setText(registered.settings.certificatFile);
	d->ignoreSslError->setChecked(registered.settings.ignoreSslError);
}

void FileConnectionDialog::slotTextChanged(const QString& text)
{
	d->remoteFileChanged(text);
}

void FileConnectionDialog::slotProtocolChanged(int index)
{
	d->protocolChanged(index);
}
