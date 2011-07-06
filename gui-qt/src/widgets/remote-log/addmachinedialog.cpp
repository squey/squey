#include <logviewer/addmachinedialog.h>

#include <QFormLayout>
#include <QVBoxLayout>
#include <QLineEdit>
#include <QLabel>
#include <QDialogButtonBox>
#include <QPushButton>

class AddMachineDialog::AddMachineDialogPrivate
{
public:
    AddMachineDialogPrivate( AddMachineDialog *q)
        :machineName( 0 ),
         hostName( 0 ),
         buttons( 0 ),
         qq( q )
    {
    }
    void initWidget();
    void machineNameChanged();
    QLineEdit *machineName;
    QLineEdit *hostName;
    QDialogButtonBox *buttons;
    AddMachineDialog *qq;
};

void AddMachineDialog::AddMachineDialogPrivate::initWidget()
{
    QVBoxLayout *layout = new QVBoxLayout;

    QFormLayout*formLayout = new QFormLayout;
    layout->addLayout( formLayout );

    machineName = new QLineEdit;
    formLayout->addRow( tr( "Machine name:" ), machineName );
    connect( machineName, SIGNAL( textChanged ( const QString & ) ), qq, SLOT( slotTextChanged() ) );


    hostName = new QLineEdit;
    formLayout->addRow( tr( "Hostname:" ), hostName );
    connect( hostName, SIGNAL( textChanged ( const QString & ) ), qq, SLOT( slotTextChanged() ) );

    buttons = new QDialogButtonBox(QDialogButtonBox::Ok|QDialogButtonBox::Cancel );
    connect( buttons, SIGNAL( accepted() ), qq, SLOT( accept() ) );
    connect( buttons, SIGNAL( rejected() ), qq, SLOT( reject() ) );
    layout->addWidget( buttons );

    machineNameChanged();
    qq->setLayout( layout );
    qq->setWindowTitle( tr( "Add new Machine" ) );

}

void AddMachineDialog::AddMachineDialogPrivate::machineNameChanged()
{
    buttons->button( QDialogButtonBox::Ok )->setEnabled( !hostName->text().isEmpty() && !machineName->text().isEmpty());
}


AddMachineDialog::AddMachineDialog( QWidget *parent )
    :QDialog( parent ), d( new AddMachineDialogPrivate(this) )
{
    d->initWidget();
    resize( 250, 100 );
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
