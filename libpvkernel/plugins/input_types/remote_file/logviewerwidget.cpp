/**
 * \file logviewerwidget.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include "include/logviewerwidget.h"
#include "include/logviewerwidget_p.h"
#include "include/fileconnectiondialog.h"
#include "include/connectionsettings.h"
#include "include/filedownloader.h"
#include "include/addmachinedialog.h"

#include <QSettings>
#include <QListWidget>
#include <QTableWidget>
#include <QLayout>
#include <QPushButton>
#include <QHeaderView>
#include <QAction>
#include <QMessageBox>
#include <QInputDialog>
#include <QFile>
#include <QDebug>

//QStringList LogViewerPrivate::defaultStringI18nProtocol = QStringList() << QObject::tr( "Local" )<<QObject::tr( "HTTP" )<<QObject::tr( "HTTPS" )<<QObject::tr( "FTP" )<<QObject::tr( "FTP over SSL" )<<QObject::tr( "SCP" )<<QObject::tr( "SFTP" );
QStringList LogViewerPrivate::defaultStringI18nProtocol = QStringList() << QObject::tr( "HTTP" )<<QObject::tr( "HTTPS" )<<QObject::tr( "FTP" )<<QObject::tr( "FTP over SSL" )<<QObject::tr( "SCP" )<<QObject::tr( "SFTP" );

class LogViewerWidget::LogViewerWidgetPrivate
{
public:
    LogViewerWidgetPrivate(LogViewerWidget *q)
        : machineListWidget( 0 ),
          filesTableWidget( 0 ),
          addFileToDownload( 0 ),
          removeFileToDownload( 0 ),
          configureFileToDownload( 0 ),
          addMachineAction( 0 ),
          removeMachineAction( 0 ),
          qq( q )
    {

    }
    enum LogViewerMachineConfigItem {
        typeMachineConfigItem = Qt::UserRole + 1
    };


    void initWidget();
    void fillList();
	void selectFirstMachine();
    void fillTableFile( const RegisteredFile& viewer );
    MachineConfig machineConfigFromItem( QListWidgetItem *item );

    QMap<MachineConfig, QList<RegisteredFile> >listOfMachine;

    QListWidget* machineListWidget;
    QTableWidget* filesTableWidget;

    FileDownLoader *fileDownLoader;

    QPushButton* addFileToDownload;
    QPushButton* removeFileToDownload;
    QPushButton* configureFileToDownload;

    QAction* addMachineAction;
    QAction* removeMachineAction;
    LogViewerWidget *qq;
};

void LogViewerWidget::LogViewerWidgetPrivate::initWidget()
{
    fileDownLoader = new FileDownLoader( qq );
    connect( fileDownLoader, SIGNAL( downloadError( const QString&, int ) ), qq, SIGNAL( downloadError( const QString&, int ) ) );

    QHBoxLayout * layout = new QHBoxLayout;
    layout->setMargin( 0 );
    machineListWidget = new QListWidget;
    layout->addWidget( machineListWidget );
    connect( machineListWidget, SIGNAL( currentItemChanged( QListWidgetItem *, QListWidgetItem * ) ), qq, SLOT( slotFillFilesList( QListWidgetItem * ) ) );

    QVBoxLayout *fileLayout = new QVBoxLayout;
    layout->addLayout( fileLayout );
    filesTableWidget = new QTableWidget;
    filesTableWidget->setColumnCount( 3 );
    filesTableWidget->setSelectionBehavior( QAbstractItemView::SelectRows );
    filesTableWidget->verticalHeader()->hide();

    QStringList headerLabel;
    headerLabel<<tr( "Path" )<<tr( "Protocol" )<<tr( "Port" );
    filesTableWidget->setHorizontalHeaderLabels( headerLabel );

    connect( filesTableWidget, SIGNAL( itemDoubleClicked ( QTableWidgetItem *) ), qq->parent(), SLOT( slotDownloadFiles() ) );
    connect( filesTableWidget, SIGNAL( itemClicked ( QTableWidgetItem * ) ), qq, SLOT( slotUpdateButtons() ) );
    fileLayout->addWidget( filesTableWidget );

    QHBoxLayout *actionLayout = new QHBoxLayout;
    fileLayout->addLayout( actionLayout );

    addFileToDownload = new QPushButton( tr( "Add File..." ) );
    connect( addFileToDownload, SIGNAL( clicked() ), qq, SLOT( slotAddFile() ) );
    actionLayout->addWidget( addFileToDownload );

    removeFileToDownload = new QPushButton( tr( "Remove File" ) );
    connect( removeFileToDownload, SIGNAL( clicked() ), qq, SLOT( slotRemoveFile() ) );
    actionLayout->addWidget( removeFileToDownload );


    configureFileToDownload = new QPushButton( tr( "Configure File" ) );
    connect( configureFileToDownload, SIGNAL( clicked() ), qq, SLOT( slotConfigureFile() ) );
    actionLayout->addWidget( configureFileToDownload );

    addMachineAction = new QAction( tr( "Add New Machine..." ), qq );
    connect( addMachineAction, SIGNAL( triggered(bool) ), qq, SLOT( slotAddMachine() ) );

    removeMachineAction = new QAction( tr( "Remove Machine" ), qq );
    connect( removeMachineAction, SIGNAL( triggered(bool) ), qq, SLOT( slotRemoveMachine() ) );


    qq->setLayout( layout );
}


void LogViewerWidget::LogViewerWidgetPrivate::fillTableFile( const RegisteredFile& viewer )
{
    const int row = filesTableWidget->rowCount();
    filesTableWidget->setRowCount( row + 1 );
    QTableWidgetItem *item0 = new QTableWidgetItem(viewer.remoteFile);
    item0->setFlags( item0->flags()&~Qt::ItemIsEditable );
    QTableWidgetItem *item1 = new QTableWidgetItem( LogViewerPrivate::defaultStringI18nProtocol[viewer.settings.protocol]);
    item1->setFlags( item1->flags()&~Qt::ItemIsEditable );
    QTableWidgetItem *item2 = new QTableWidgetItem(QString::number( viewer.settings.port ));
    item2->setFlags( item2->flags()&~Qt::ItemIsEditable );
    filesTableWidget->setItem( row, 0, item0 );
    filesTableWidget->setItem( row, 1, item1 );
    filesTableWidget->setItem( row, 2, item2 );
}


void LogViewerWidget::LogViewerWidgetPrivate::fillList()
{
    machineListWidget->clear();
    const QList<MachineConfig> lst = listOfMachine.keys();
    Q_FOREACH( const MachineConfig& machine, lst ) {
        QListWidgetItem *item  = new QListWidgetItem( machine.name );
        item->setData( typeMachineConfigItem , machine.hostname );
        machineListWidget->addItem( item );
    }
}

void LogViewerWidget::LogViewerWidgetPrivate::selectFirstMachine()
{
	if (machineListWidget->count() <= 0) {
		return;
	}

	QListWidgetItem* first = machineListWidget->item(0);
	machineListWidget->setCurrentItem(first);
}

MachineConfig LogViewerWidget::LogViewerWidgetPrivate::machineConfigFromItem( QListWidgetItem *item )
{
    if ( item )
    {
        const QString machineName = item->text();
        const QString hostName = item->data( LogViewerWidgetPrivate::typeMachineConfigItem ).toString();
        MachineConfig machineConfig( machineName, hostName );
        return machineConfig;
    }
    return MachineConfig();
}



LogViewerWidget::LogViewerWidget( QWidget * _parent )
    :QWidget( _parent ), d( new LogViewerWidgetPrivate( this ) )
{
    d->initWidget();
    loadSettings();

    d->fillList();
	d->selectFirstMachine();

    slotUpdateButtons();
}

LogViewerWidget::~LogViewerWidget()
{
    saveSettings();
    delete d;
}

void LogViewerWidget::saveSettings()
{
    QSettings settings( QLatin1String( "Picviz" ), QLatin1String( "logviewerwidget" ) );
    settings.beginGroup( QLatin1String( "Machine" ) );
    //Be sure to clear it
    settings.clear();
    QMapIterator<MachineConfig, QList<RegisteredFile> > i( d->listOfMachine );
    while (i.hasNext()) {
        i.next();
        settings.beginGroup( i.key().name );
        settings.setValue( QString::fromLatin1( "HostName" ), i.key().hostname );

        const QList<RegisteredFile> registeredfiles = i.value();
        int value = 0;
        Q_FOREACH( const RegisteredFile& registered, registeredfiles )
        {
            settings.beginGroup( QString::fromLatin1( "File%1" ).arg( value ) );
            settings.setValue( QLatin1String( "RemoteFile" ), registered.remoteFile );
            settings.setValue( QLatin1String( "LocalFile" ), registered.localFile );
            settings.setValue( QLatin1String( "SshKeyFile" ), saveSshKeyFile( registered.settings.sshKeyFile ) );
            settings.setValue( QLatin1String( "Password" ), encryptPassword( registered.settings.password ) );
            settings.setValue( QLatin1String( "Login" ), registered.settings.login );
            settings.setValue( QLatin1String( "Port" ), registered.settings.port );
            settings.setValue( QLatin1String( "Protocol" ), registered.settings.protocol );
            settings.setValue( QLatin1String( "CertificatFile" ), saveCertificateFile( registered.settings.certificatFile ) );
            settings.setValue( QLatin1String( "IgnoreSslError" ), registered.settings.ignoreSslError );

            settings.endGroup();
            value++;
        }
        settings.endGroup();
    }
    settings.endGroup();
    settings.sync();
}

void LogViewerWidget::loadSettings()
{
    QSettings settings( QLatin1String( "Picviz" ), QLatin1String( "logviewerwidget" ) );
    settings.beginGroup( QLatin1String( "Machine" ) );
    const QStringList list = settings.childGroups();
    const int countList = list.count();
    for ( int i = 0; i< countList; ++i )
    {
        settings.beginGroup( list[i] );
        const QString hostname = settings.value( QLatin1String( "HostName" ) ).toString();

        QList<RegisteredFile> lstRegister;
        const QStringList childlist = settings.childGroups();
        const int childCount = childlist.count();
        for ( int subGroup = 0; subGroup < childCount; ++subGroup )
        {
            settings.beginGroup( childlist[subGroup] );
            RegisteredFile registered;
            registered.remoteFile = settings.value( QLatin1String( "RemoteFile" ) ).toString();
            registered.localFile = settings.value( QLatin1String( "LocalFile" ) ).toString();
            registered.settings.sshKeyFile = loadSshKeyFile( settings.value( QLatin1String( "SshKeyFile" ) ).toString() );
            registered.settings.password = unencryptPassword( settings.value( QLatin1String( "Password" ) ).toString() );
            registered.settings.login = settings.value( QLatin1String( "Login" ) ).toString();
            registered.settings.port = settings.value( QLatin1String( "Port" ) ).toInt();
            registered.settings.protocol = static_cast<Protocol>( settings.value( QLatin1String( "Protocol" ) ).toInt() );
            registered.settings.certificatFile = loadCertificateFile( settings.value( QLatin1String( "CertificatFile" ) ).toString() );
            registered.settings.ignoreSslError = settings.value( QLatin1String( "IgnoreSslError" ) ).toBool();

            settings.endGroup();
            lstRegister<<registered;
        }
        MachineConfig machineConfig( list[i], hostname );
        d->listOfMachine.insert( machineConfig, lstRegister );

        settings.endGroup();
    }
}

QAction *LogViewerWidget::addMachineAction() const
{
    return d->addMachineAction;
}

QAction *LogViewerWidget::removeMachineAction() const
{
    return d->removeMachineAction;
}

void LogViewerWidget::slotAddFile()
{
    if ( !d->machineListWidget->currentItem() ) //be safe
        return;

    MachineConfig machineConfig = d->machineConfigFromItem( d->machineListWidget->currentItem() );

    FileConnectionDialog dialog( this );
    if ( d->filesTableWidget->rowCount() > 0 )
    {
        RegisteredFile registered = d->listOfMachine.value( machineConfig ).at( 0 );
        dialog.initialize( registered, machineConfig.hostname );
    }

    if ( dialog.exec() ) {
        const RegisteredFile registered = dialog.registeredFileSettings();
        QList<RegisteredFile>& listFiles = d->listOfMachine[ machineConfig ];
        listFiles.append( registered );
        slotFillFilesList( d->machineListWidget->currentItem() );
        slotUpdateButtons();
    }
}

void LogViewerWidget::slotRemoveFile()
{
    if ( !d->machineListWidget->currentItem() ) //be safe
        return;
    if ( QMessageBox::warning( this, tr( "Remove File?" ), tr( "Do you want to remove this file ?" ), QMessageBox::Ok|QMessageBox::Cancel ) == QMessageBox::Cancel )
        return;

    MachineConfig machineConfig = d->machineConfigFromItem( d->machineListWidget->currentItem() );

    QList<RegisteredFile>& listFiles = d->listOfMachine[ machineConfig ];
    listFiles.removeAt( d->filesTableWidget->currentRow() );

    slotFillFilesList( d->machineListWidget->currentItem() );

    slotUpdateButtons();
}

void LogViewerWidget::slotConfigureFile()
{
    if ( !d->filesTableWidget->currentItem() ) //Be safe
        return;
    FileConnectionDialog dialog( this );
    const int row = d->filesTableWidget->currentRow();

    MachineConfig machineConfig = d->machineConfigFromItem( d->machineListWidget->currentItem() );

    RegisteredFile registered = d->listOfMachine.value( machineConfig ).at( row );
    dialog.initialize( registered, machineConfig.hostname );
    if ( dialog.exec() ) {
        QList<RegisteredFile>& lstRegistered = d->listOfMachine[ machineConfig ];
        lstRegistered[row] = dialog.registeredFileSettings();
        slotFillFilesList( d->machineListWidget->currentItem() );
    }
}


void LogViewerWidget::slotFillFilesList( QListWidgetItem *item )
{
    if ( !item )
        return;
    d->filesTableWidget->clearContents();
    d->filesTableWidget->setRowCount( 0 );

    MachineConfig machineConfig = d->machineConfigFromItem( item );

    const QList<RegisteredFile> listFiles = d->listOfMachine.value( machineConfig );
    Q_FOREACH( const RegisteredFile& registered, listFiles ) {
        d->fillTableFile( registered );
    }
    slotUpdateButtons();
}

bool LogViewerWidget::downloadSelectedFiles(QHash<QString, QUrl>& dl_files)
{
    MachineConfig machineConfig = d->machineConfigFromItem( d->machineListWidget->currentItem() );

    QList<RegisteredFile>& lstRegistered = d->listOfMachine[ machineConfig ];

	// Get the indexes of the selected files
	QList<QTableWidgetItem*> sel_files = d->filesTableWidget->selectedItems();
	// And download them one by one
	bool ret = false;
	for (int i = 0; i < sel_files.size(); i++) {
		QTableWidgetItem* item = sel_files[i];
		if (item->column() != 0) {
			continue;
		}
		RegisteredFile& registered = lstRegistered[item->row()];

		QString temporaryFilePath;
		QString errorMessage;
		QUrl url;
		bool cancel = false;
		const bool res = d->fileDownLoader->download( registered.remoteFile, temporaryFilePath, registered.settings, machineConfig.hostname, errorMessage, url, cancel );
		if ( res ) {
			registered.localFile = temporaryFilePath;
			//qDebug()<<" temporaryFilePath :"<<temporaryFilePath;
			emit newFile( machineConfig.name, registered.remoteFile, temporaryFilePath);
			dl_files[temporaryFilePath] = url;
			ret = true;
		} else {
			if (!cancel) {
				QMessageBox::critical( this, tr( "Error" ), errorMessage.isEmpty() ? tr( "Can not initialize download from libcurl" ) : errorMessage );
			}
			continue;
		}
	}
	return ret;
}

void LogViewerWidget::slotUpdateButtons()
{
    const bool itemSelected = ( d->filesTableWidget->currentItem() != 0 );
    d->removeFileToDownload->setEnabled( itemSelected );
    d->configureFileToDownload->setEnabled( itemSelected );
    d->addFileToDownload->setEnabled( d->machineListWidget->currentItem() );
    d->removeMachineAction->setEnabled( d->machineListWidget->currentItem() );
}

void LogViewerWidget::slotAddMachine()
{
    AddMachineDialog dialog( this );
    if ( dialog.exec() )
    {
        const QString name = dialog.machineName();
        const QString hostname = dialog.hostName();
        MachineConfig machineConfig( name,hostname );


        if ( d->listOfMachine.contains( machineConfig ) )
        {
            QMessageBox::critical( this, tr( "Add Machine" ), tr( "This name is already used. We can not add it." ) );
        }
        else
        {
            d->listOfMachine.insert( machineConfig, QList<RegisteredFile>() );
            d->fillList();
        }

    }
}

void LogViewerWidget::slotRemoveMachine()
{
    if ( !d->machineListWidget->currentItem() )
        return;
    if ( QMessageBox::warning( this, tr( "Remove Machine?" ), tr( "Do you want to remove this Machine ?" ), QMessageBox::Ok|QMessageBox::Cancel ) == QMessageBox::Cancel )
        return;

    MachineConfig machineConfig = d->machineConfigFromItem( d->machineListWidget->currentItem() );

    d->listOfMachine.remove( machineConfig );
    d->filesTableWidget->clearContents();
    d->filesTableWidget->setRowCount( 0 );

    d->fillList();
    slotUpdateButtons();
}

void LogViewerWidget::removeLocalFile( const QString& localFile)
{
    QFile file( localFile );
    if ( file.exists() )
    {
        const bool res = file.remove();
        if ( !res )
        {
            QMessageBox::critical( this, tr( "Remove Local File" ), tr( "Can not remove \"%1\"" ).arg( localFile ) );
        }

    }
}

QString LogViewerWidget::encryptPassword( const QString& password )
{
    //TODO add method to encrypt password
    return password;
}

QString LogViewerWidget::unencryptPassword( const QString& password )
{
    //TODO add method to unencrypt password
    return password;
}


QString LogViewerWidget::loadSshKeyFile( const QString& sshKeyFile )
{
    return sshKeyFile;
}

QString LogViewerWidget::saveSshKeyFile( const QString& sshKeyFile )
{
    return sshKeyFile;
}

QString LogViewerWidget::loadCertificateFile( const QString& certificateFile )
{
    return certificateFile;
}

QString LogViewerWidget::saveCertificateFile( const QString& certificateFile )
{
    return certificateFile;
}


QString LogViewerWidget::authentication( const QString& machineName, const QString& filename )
{
    const QList<MachineConfig> lst = d->listOfMachine.keys();
    Q_FOREACH( const MachineConfig& machine, lst ) {
        if ( machineName == machine.name )
        {
            QList<RegisteredFile>& lstRegistered = d->listOfMachine[ machine ];
            if ( !lstRegistered.isEmpty() )
            {
                Q_FOREACH( const RegisteredFile& registered, lstRegistered )
                {
                    if ( registered.remoteFile == filename )
                    {
                        if ( !registered.settings.password.isEmpty() )
                        {
                            return tr( "Authentication by password" );
                        }
                        else if ( !registered.settings.sshKeyFile.isEmpty() )
                        {
                            return tr( "Authentication by SSH key" );
                        }
                    }
                }
            }

        }
    }
    return QString();
}
