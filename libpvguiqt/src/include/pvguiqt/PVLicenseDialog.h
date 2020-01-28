/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2018
 */

#ifndef __PVGUIQT_PVLICENSEDIALOG_H__
#define __PVGUIQT_PVLICENSEDIALOG_H__

#include <QMessageBox>
#include <QApplication>
#include <QClipboard>
#include <QSpacerItem>
#include <QDir>
#include <QFile>
#include <QFontMetrics>
#include <QRadioButton>
#include <QButtonGroup>
#include <QComboBox>

#include <pvkernel/core/PVConfig.h>
#include <pvkernel/core/PVLicenseActivator.h>
#include <pvkernel/widgets/PVFileDialog.h>

namespace PVGuiQt
{

class PVLicenseDialog : public QDialog
{
  public:
	PVLicenseDialog(const QString& inendi_license_path,
	                const QString& product,
	                const QString& title = {},
	                const QString& text = {},
	                const QIcon& icon = {},
	                QWidget* parent = nullptr)
	    : QDialog(parent)
	{
		QString locking_code =
		    QString::fromStdString(PVCore::PVLicenseActivator::get_locking_code());

		QString host_id = QString::fromStdString(PVCore::PVLicenseActivator::get_host_id());

		// Title
		setWindowTitle(title);

		// Icon
		QVBoxLayout* icon_layout = new QVBoxLayout();
		QLabel* icon_label = new QLabel();
		QPixmap pixmap = icon.pixmap(QSize(32, 32));
		icon_label->setPixmap(pixmap);
		icon_layout->addWidget(icon_label);
		icon_layout->addSpacerItem(
		    new QSpacerItem(1, 1, QSizePolicy::Minimum, QSizePolicy::Expanding));

		// Main text
		_text = new QLabel(text);

		// Dialog buttons
		QDialogButtonBox* buttons =
		    new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
		buttons->button(QDialogButtonBox::Ok)->setEnabled(false);

		// Online activation
		QHBoxLayout* online_layout = new QHBoxLayout();
		QRadioButton* online_radiobutton = new QRadioButton();
		QGroupBox* online_groupbox = new QGroupBox("Online activation");
		online_layout->addWidget(online_radiobutton);
		online_layout->addWidget(online_groupbox);
		QVBoxLayout* online_groupbox_layout = new QVBoxLayout();

		QLabel* license_type_label = new QLabel("License type:");
		QComboBox* license_type_combobox = new QComboBox();
		QHBoxLayout* license_type_layout = new QHBoxLayout();
		license_type_layout->addWidget(license_type_label);
		license_type_layout->addWidget(license_type_combobox);
		QHBoxLayout* license_token_layout = new QHBoxLayout();
		QLabel* token_label = new QLabel();
		QLineEdit* token_text = new QLineEdit();
		license_token_layout->addWidget(token_label);
		license_token_layout->addWidget(token_text);
		connect(license_type_combobox,
		        static_cast<void (QComboBox::*)(const QString&)>(&QComboBox::currentIndexChanged),
		        [=](const QString& text) {
			        assert(text == "Trial" or text == "Paid");
			        if (text == "Trial") { // Prompt validated email
				        token_label->setText("Validated email:");
				        QRegularExpression email_rx("\\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\\.[A-Z]{2,4}\\b",
				                                    QRegularExpression::CaseInsensitiveOption);
				        token_text->setValidator(new QRegularExpressionValidator(email_rx, this));
			        } else { // Prompt activaton code
				        token_label->setText("Activation key:");
				        QRegularExpression activation_code_rx("\\b[A-Za-z0-9]{16}\\b");
				        token_text->setValidator(
				            new QRegularExpressionValidator(activation_code_rx, this));
			        }
			        token_text->setText("");
			        online_radiobutton->setChecked(true);
		        });
		license_type_combobox->addItems(QStringList{"Trial", "Paid"});
		auto check_email_valid_f = [=]() {
			if (online_radiobutton->isChecked()) {
				buttons->button(QDialogButtonBox::Ok)->setEnabled(token_text->hasAcceptableInput());
			}
			online_radiobutton->setChecked(true);
		};
		connect(token_text, &QLineEdit::textChanged, check_email_valid_f);
		connect(online_radiobutton, &QRadioButton::clicked, check_email_valid_f);
		online_groupbox_layout->addLayout(license_type_layout);
		online_groupbox_layout->addLayout(license_token_layout);
		online_groupbox->setLayout(online_groupbox_layout);

		// Offline activation
		QHBoxLayout* offline_layout = new QHBoxLayout();
		QRadioButton* offline_radiobutton = new QRadioButton();
		offline_radiobutton->setChecked(true);
		connect(offline_radiobutton, &QRadioButton::clicked, [=]() {
			buttons->button(QDialogButtonBox::Ok)->setEnabled(not _user_license_path.isEmpty());
		});
		QGroupBox* offline_groupbox = new QGroupBox("Offline activation");
		QVBoxLayout* offline_groupbox_layout = new QVBoxLayout();
		offline_groupbox->setLayout(offline_groupbox_layout);
		offline_layout->addWidget(offline_radiobutton);
		offline_layout->addWidget(offline_groupbox);

		// locking_code
		QHBoxLayout* locking_code_layout = new QHBoxLayout;
		QLineEdit* locking_code_text = new QLineEdit(locking_code);
		locking_code_text->setFixedWidth(
		    QFontMetrics(locking_code_text->font()).width(locking_code + "  "));
		locking_code_text->setFocusPolicy(Qt::NoFocus);
		QPushButton* copy_locking_code_button = new QPushButton();
		connect(copy_locking_code_button, &QPushButton::clicked,
		        [=]() { QApplication::clipboard()->setText(locking_code); });
		copy_locking_code_button->setToolTip("Copy to clipboard");
		copy_locking_code_button->setIcon(QIcon(":/edit-copy.png"));
		copy_locking_code_button->setFocusPolicy(Qt::NoFocus);
		copy_locking_code_button->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
		locking_code_text->setReadOnly(true);
		locking_code_layout->addWidget(new QLabel("Locking code: "));
		locking_code_layout->addWidget(locking_code_text);
		locking_code_layout->addWidget(copy_locking_code_button);

		// host_id
		QHBoxLayout* host_id_layout = new QHBoxLayout;
		QLineEdit* host_id_text = new QLineEdit(host_id);
		host_id_text->setFixedWidth(QFontMetrics(host_id_text->font()).width(host_id + "  "));
		host_id_text->setFocusPolicy(Qt::NoFocus);
		QPushButton* copy_host_id_button = new QPushButton();
		connect(copy_host_id_button, &QPushButton::clicked,
		        [=]() { QApplication::clipboard()->setText(host_id); });
		copy_host_id_button->setToolTip("Copy to clipboard");
		copy_host_id_button->setIcon(QIcon(":/edit-copy.png"));
		copy_host_id_button->setFocusPolicy(Qt::NoFocus);
		copy_host_id_button->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
		host_id_text->setReadOnly(true);
		host_id_layout->addWidget(new QLabel("Host ID: "));
		host_id_layout->addWidget(host_id_text);
		host_id_layout->addWidget(copy_host_id_button);

		QHBoxLayout* license_file_layout = new QHBoxLayout;
		QPushButton* browse_license_file_button = new QPushButton("...");
		browse_license_file_button->adjustSize();
		browse_license_file_button->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
		connect(browse_license_file_button, &QPushButton::clicked, [=]() {
			_user_license_path = PVWidgets::PVFileDialog::getOpenFileName(
			    this, "Browse your license file", "", QString("License file (*.lic)"));
			offline_radiobutton->setChecked(not _user_license_path.isEmpty());
			buttons->button(QDialogButtonBox::Ok)->setEnabled(not _user_license_path.isEmpty());
		});
		license_file_layout->addWidget(new QLabel("Browse license file: "));
		license_file_layout->addSpacerItem(
		    new QSpacerItem(1, 1, QSizePolicy::Expanding, QSizePolicy::Minimum));
		license_file_layout->addWidget(browse_license_file_button);

		QHBoxLayout* license_server_layout = new QHBoxLayout;
		license_server_layout->addWidget(new QLabel("License server: "));

		QLineEdit* license_server_line_edit = new QLineEdit("");
		license_server_line_edit->setToolTip("port@license-server");
		QRegularExpression server_rx("\\b[0-9]{2,5}@[A-Z0-9.-]+\\b",
		                             QRegularExpression::CaseInsensitiveOption);
		license_server_line_edit->setValidator(new QRegularExpressionValidator(server_rx, this));
		auto check_server_valid_f = [=]() {
			if (offline_radiobutton->isChecked()) {
				buttons->button(QDialogButtonBox::Ok)
				    ->setEnabled(license_server_line_edit->hasAcceptableInput());
			}
			offline_radiobutton->setChecked(true);
		};
		connect(license_server_line_edit, &QLineEdit::textChanged, check_server_valid_f);
		license_server_layout->addWidget(license_server_line_edit);

		offline_groupbox_layout->addLayout(locking_code_layout);
		if (product != "pcap-inspector") {
			offline_groupbox_layout->addLayout(host_id_layout);
		}
		offline_groupbox_layout->addLayout(license_file_layout);
		if (product != "pcap-inspector") {
			offline_groupbox_layout->addLayout(license_server_layout);
		}

		QButtonGroup* radiobutton_group = new QButtonGroup(this);
		radiobutton_group->addButton(online_radiobutton, 0);
		radiobutton_group->addButton(offline_radiobutton, 1);
		QVBoxLayout* content_layout = new QVBoxLayout;
		content_layout->addWidget(_text);
		if (product ==
		    "pcap-inspector") { // Enable online activation only for pcap-inspector for now
			content_layout->addLayout(online_layout);
			online_radiobutton->setChecked(true);
		}
		content_layout->addLayout(offline_layout);
		content_layout->addSpacerItem(
		    new QSpacerItem(1, 1, QSizePolicy::Minimum, QSizePolicy::Expanding));
		content_layout->addWidget(buttons);

		QHBoxLayout* layout = new QHBoxLayout;
		layout->addLayout(icon_layout);
		layout->addLayout(content_layout);

		setLayout(layout);

		connect(buttons, &QDialogButtonBox::accepted, this, [=]() {
			PVCore::PVLicenseActivator::EError err_code;
			if (offline_radiobutton->isChecked()) {
				if (QFileInfo(_user_license_path).exists()) {
					err_code = offline_activation(_user_license_path, inendi_license_path);
				} else { // use license server
					PVCore::PVLicenseActivator::license_server() =
					    license_server_line_edit->text().toStdString();
					err_code = PVCore::PVLicenseActivator::EError::NO_ERROR;
				}
			} else {
				std::string email = token_text->text().toStdString();
				std::string activation_key = token_text->text().toStdString();
				if (license_type_combobox->currentText() == "Trial") {
					activation_key = std::string("demo_") + product.toStdString();
				} else { // Paid
					email = "";
				}
				err_code = online_activation(email, locking_code.toStdString(), activation_key,
				                             inendi_license_path);
			}

			if (err_code != PVCore::PVLicenseActivator::EError::NO_ERROR) {
				QMessageBox::warning(this, _error_strings.at(err_code).first,
				                     _error_strings.at(err_code).second, QMessageBox::Ok);
				return;
			}

			accept();
		});
		connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);
	}

  public:
	static bool show_no_license(const QString& inendi_license_path, const QString& product)
	{
		PVLicenseDialog dlg(inendi_license_path, product, "Software activation",
		                    "Please, activate the software.                              <br>",
		                    QApplication::style()->standardIcon(QStyle::SP_MessageBoxInformation));
		return dlg.exec() == QDialog::Accepted;
	}

	static bool show_license_expired(const QString& inendi_license_path, const QString& product)
	{
		PVLicenseDialog dlg(inendi_license_path, product, "License expired",
		                    "The license has expired. Please, re-activate the software.<br>",
		                    QApplication::style()->standardIcon(QStyle::SP_MessageBoxWarning));
		return dlg.exec() == QDialog::Accepted;
	}

	static bool show_bad_license(const QString& inendi_license_path, const QString& product)
	{
		PVLicenseDialog dlg(inendi_license_path, product, "Hardware identification error",
		                    "This license does not allow you to run the software on this "
		                    "hardware.<br>Please, check you are using the proper license.<br>",
		                    QApplication::style()->standardIcon(QStyle::SP_MessageBoxCritical));
		return dlg.exec() == QDialog::Accepted;
	}

	static bool show_memory_exceeded(const QString& inendi_license_path, const QString& product)
	{
		PVLicenseDialog dlg(
		    inendi_license_path, product, "Maximum authorized memory exceeded",
		    "This license does not allow you to run the software on a machine<br>"
		    "with such a large amount of memory.<br><br>Please, upgrade your license.<br>",
		    QApplication::style()->standardIcon(QStyle::SP_MessageBoxCritical));
		return dlg.exec() == QDialog::Accepted;
	}

	static bool show_unable_to_contact_server_error(const QString& inendi_license_path,
	                                                const QString& product)
	{
		PVLicenseDialog dlg(inendi_license_path, product, "Unable to contact license server",
		                    "Please check that your license server is properly configured<br>"
		                    "and accessible.<br>",
		                    QApplication::style()->standardIcon(QStyle::SP_MessageBoxCritical));
		return dlg.exec() == QDialog::Accepted;
	}

	static bool show_license_unknown_error(const QString& inendi_license_path,
	                                       const QString& product)
	{
		PVLicenseDialog dlg(inendi_license_path, product, "License error", "Unkown license error.",
		                    QApplication::style()->standardIcon(QStyle::SP_MessageBoxCritical));
		return dlg.exec() == QDialog::Accepted;
	}

  private:
	PVCore::PVLicenseActivator::EError offline_activation(const QString& user_license_path,
	                                                      const QString& inendi_license_path)
	{
		return PVCore::PVLicenseActivator(inendi_license_path.toStdString())
		    .offline_activation(user_license_path);
	}

	PVCore::PVLicenseActivator::EError online_activation(const std::string& email,
	                                                     const std::string& locking_code,
	                                                     const std::string& activation_key,
	                                                     const QString& inendi_license_path)
	{
		return PVCore::PVLicenseActivator(inendi_license_path.toStdString())
		    .online_activation(email, locking_code, activation_key);
	}

  private:
	const std::unordered_map<PVCore::PVLicenseActivator::EError, std::pair<QString, QString>>
	    _error_strings{
	        {PVCore::PVLicenseActivator::EError::NO_INTERNET_CONNECTION,
	         {"Error when reaching activation service", "An error occured when trying to contact "
	                                                    "the activation service. Please check your "
	                                                    "Internet connection."}},
	        {PVCore::PVLicenseActivator::EError::ACTIVATION_SERVICE_UNAVAILABLE,
	         {"Activation service unavailable", "Our activation service in currenly offline for "
	                                            "maintenance. Plase try again later."}},
	        {PVCore::PVLicenseActivator::EError::UNABLE_TO_READ_LICENSE_FILE,
	         {"Unable to read license file", "Please check your license read permissions."}},
	        {PVCore::PVLicenseActivator::EError::UNABLE_TO_INSTALL_LICENSE_FILE,
	         {"Unable to install license file",
	          "Please check your '.inendi' folder write permissions."}},
	        {PVCore::PVLicenseActivator::EError::UNKOWN_USER,
	         {"Unknown email address",
	          "Please check that your email address was registered online first."}},
	        {PVCore::PVLicenseActivator::EError::USER_NOT_VALIDATED,
	         {"Email address not validated", "Your email address is registered but not validated "
	                                         "yet. Please validate your email address."}},
	        {PVCore::PVLicenseActivator::EError::TRIAL_ALREADY_ACTIVATED,
	         {"Trial already activated",
	          "This trial version is already activated on another computer."}},
	        {PVCore::PVLicenseActivator::EError::UNKNOWN_ACTIVATION_KEY,
	         {"Invalid activation key", "The activation key you provided is not valid"}},
	        {PVCore::PVLicenseActivator::EError::ACTIVATION_KEY_ALREADY_ACTIVATED,
	         {"Activation key already used",
	          "The activation key you provided is already activated on another computer."}}};

  private:
	QLabel* _text;
	QString _user_license_path;
};

} // namespace PVGuiQt

#endif // __PVGUIQT_PVLICENSEDIALOG_H__
