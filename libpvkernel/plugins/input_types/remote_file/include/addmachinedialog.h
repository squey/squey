#ifndef ADDMACHINEDIALOG_H
#define ADDMACHINEDIALOG_H

#include <QDialog>

class AddMachineDialog : public QDialog
{
    Q_OBJECT
public:
    explicit AddMachineDialog( QWidget *parent );
    ~AddMachineDialog();

    QString machineName() const;
    QString hostName() const;

private Q_SLOTS:
    void slotTextChanged();

private:
    class AddMachineDialogPrivate;
    AddMachineDialogPrivate* d;
};

#endif /* ADDMACHINEDIALOG_H */

