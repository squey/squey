#ifndef PVGUIQT_PVCREDENTIALDIALOG_H
#define PVGUIQT_PVCREDENTIALDIALOG_H

#include <QDialog>
#include <QString>
#include <QLineEdit>

namespace PVGuiQt
{
class CredentialDialog : public QDialog
{
  public:
	CredentialDialog(QWidget* parent = nullptr);

	QString get_login() const { return _login.text(); }
	QString get_password() const { return _login.text(); }

  private:
	QLineEdit _login;
	QLineEdit _passwd;
};

} // namespace PVGuiQt

#endif
