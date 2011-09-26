#include "include/filedownloader.h"
#include "include/logviewer_config.h"
#include <curl/curl.h>

#include <QTemporaryFile>
#include <QDir>
#include <QUrl>
#include <QDebug>


class FileDownLoader::FileDownLoaderPrivate
{
public:
    FileDownLoaderPrivate()
        : curl( 0 ),
          tempFile( 0 )
#ifdef ADD_DEBUG_TO_FILE
        , debugTempFile( 0 )
#endif
    {
    }
    void initialize();
    void cleanup();
    void initializeDownload( const QString &remoteFile, const ConnectionSettings& settings, const QString& hostName );
    void initializeEncrypted( const ConnectionSettings& settings );
    void initializeSsl( const ConnectionSettings& settings );
    static size_t writeData(void *buffer, size_t size, size_t nmemb, void *stream);
#ifdef ADD_DEBUG_TO_FILE
    static int writeDebugToFile(CURL *, curl_infotype, char *, size_t, void *);
#endif
    CURL *curl;
    QTemporaryFile *tempFile;
#ifdef ADD_DEBUG_TO_FILE
    QTemporaryFile *debugTempFile;
#endif
};

void FileDownLoader::FileDownLoaderPrivate::initialize()
{
    curl_global_init(CURL_GLOBAL_DEFAULT);
}

void FileDownLoader::FileDownLoaderPrivate::cleanup()
{
    if ( tempFile ) {
        tempFile->close();
        tempFile->deleteLater();
        tempFile = 0;
    }
#ifdef ADD_DEBUG_TO_FILE
    if ( debugTempFile ) {
        debugTempFile->close();
        debugTempFile->deleteLater();
        debugTempFile = 0;
    }
#endif
    /* always cleanup */
    if ( curl )
        curl_easy_cleanup(curl);
}


void FileDownLoader::FileDownLoaderPrivate::initializeDownload( const QString &remoteFile, const ConnectionSettings& settings, const QString& hostName )
{
    //static const char* s_schemes[] = { "file", "http", "https", "ftp", "ftps","scp", "sftp" };
    static const char* s_schemes[] = { "http", "https", "ftp", "ftps","scp", "sftp", "file" };
    QUrl url;
    if( settings.protocol == Local )
    {
        url= QUrl::fromLocalFile( remoteFile );
    }
    else
    {
        const QString scheme = QString::fromLatin1(s_schemes[settings.protocol]);
        url.setScheme(scheme);
        url.setHost(hostName);
        url.setPath(remoteFile);
        if ( settings.port != 0 ) {
            curl_easy_setopt( curl, CURLOPT_PORT, settings.port );
        }

        if ( settings.protocol == Https || settings.protocol == Ftps ) {
            initializeSsl( settings );
            initializeEncrypted( settings );
        }

        if ( settings.protocol == Ftps || settings.protocol == Scp || settings.protocol == SFtp ) {
            initializeEncrypted( settings );
        }
    }
    //qDebug()<<" url :"<<url;
    curl_easy_setopt(curl, CURLOPT_URL, url.toEncoded().constData() );
}

void FileDownLoader::FileDownLoaderPrivate::initializeSsl( const ConnectionSettings& settings )
{
    if ( settings.ignoreSslError )
    {
        /*
         * If you want to connect to a site who isn't using a certificate that is
         * signed by one of the certs in the CA bundle you have, you can skip the
         * verification of the server's certificate. This makes the connection
         * A LOT LESS SECURE.
         *
         * If you have a CA cert for the server stored someplace else than in the
         * default bundle, then the CURLOPT_CAPATH option might come handy for
         * you.
         */
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
        /*
         * If the site you're connecting to uses a different host name that what
         * they have mentioned in their server certificate's commonName (or
         * subjectAltName) fields, libcurl will refuse to connect. You can skip
         * this check, but this will make the connection less secure.
         */
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);
    }
    if ( !settings.certificatFile.isEmpty() )
    {
        /* cert is stored PEM coded in file... */
        /* since PEM is default, we needn't set it for PEM */
        curl_easy_setopt(curl,CURLOPT_SSLCERTTYPE,"PEM");

        /* set the cert for client authentication */
        curl_easy_setopt(curl,CURLOPT_SSLCERT,settings.certificatFile.toLatin1().constData() );
    }
}


void FileDownLoader::FileDownLoaderPrivate::initializeEncrypted( const ConnectionSettings& settings )
{
    if ( settings.password.isEmpty() )
    {
        if ( !settings.sshKeyFile.isEmpty() )
        {
            curl_easy_setopt( curl,CURLOPT_USERNAME, settings.login.toLocal8Bit().constData() );
            //Use ssh key
            curl_easy_setopt( curl, CURLOPT_SSH_PUBLIC_KEYFILE, QFile::encodeName(settings.sshKeyFile ).constData() );
        }
    }
    else
    {
        //Use password
        curl_easy_setopt( curl,CURLOPT_USERNAME, settings.login.toLocal8Bit().constData() );
        curl_easy_setopt( curl,CURLOPT_PASSWORD, settings.password.toLocal8Bit().constData() );
    }
}


size_t FileDownLoader::FileDownLoaderPrivate::writeData(void *buffer, size_t /*size*/, size_t nmemb, void *stream)
{
    QTemporaryFile *downloadFile = static_cast<QTemporaryFile*>( stream );
    if ( !downloadFile )
    {
        return -1; /*failure*/
    }
    const qint64 d = downloadFile->write( ( const char * )buffer, nmemb );
    downloadFile->flush();
    return d;
}

#ifdef ADD_DEBUG_TO_FILE
int FileDownLoader::FileDownLoaderPrivate::writeDebugToFile(CURL *, curl_infotype type , char *data , size_t size, void *stream)
{
    QTemporaryFile *downloadFile = static_cast<QTemporaryFile*>( stream );
    QString text;
    switch (type) {
    case CURLINFO_TEXT:
        text = tr("== Info: \n");
        break;
    case CURLINFO_HEADER_OUT:
        text = tr( "=> Send header\n" );
        break;
    case CURLINFO_DATA_OUT:
        text = tr( "=> Send data\n" );
        break;
    case CURLINFO_SSL_DATA_OUT:
        text = tr( "=> Send SSL data\n" );
        break;
    case CURLINFO_HEADER_IN:
        text = tr( "<= Recv header\n" );
        break;
    case CURLINFO_DATA_IN:
        text = tr( "<= Recv data\n" );
        break;
    case CURLINFO_SSL_DATA_IN:
        text = tr( "<= Recv SSL data\n" );
        break;
    default:
        return 0;
    }
    /*const qint64 d = */downloadFile->write( ( const char * )data, size );
    downloadFile->flush();
    return 0;
}
#endif


FileDownLoader::FileDownLoader(QObject *_parent)
    :QObject( _parent ), d( new FileDownLoaderPrivate() )
{
    d->initialize();
}

FileDownLoader::~FileDownLoader()
{
    curl_global_cleanup();
    delete d;
}

bool FileDownLoader::download( const QString &remoteFile, QString&tempFile, const ConnectionSettings& settings, const QString& hostName, QString&errorMessage )
{
    bool res = false;
    d->curl = curl_easy_init();
    if ( d->curl )
    {
        d->initializeDownload( remoteFile, settings, hostName );
        /* Define our callback to get called when there's data to be written */
        curl_easy_setopt(d->curl, CURLOPT_WRITEFUNCTION, d->writeData);

        /* Set a pointer to our struct to pass to the callback */
        d->tempFile = new QTemporaryFile();
        d->tempFile->setAutoRemove( false );
        d->tempFile->setFileTemplate( TEMPORARYFILENAME_TEMPLATE );
        res = d->tempFile->open();
        if ( !res ) {
            errorMessage = tr( "Can not open temporary file" );
            d->cleanup();
            return false;

        }
        const QString tempFileName = d->tempFile->fileName();
        qDebug()<<" d->tempFile->filename();"<<d->tempFile->fileName();

        curl_easy_setopt(d->curl, CURLOPT_WRITEDATA, d->tempFile);

#ifdef ADD_CURL_DEBUG
        /* Switch on full protocol/debug output */
        curl_easy_setopt(d->curl, CURLOPT_VERBOSE, 1L);
#ifdef ADD_DEBUG_TO_FILE
        d->debugTempFile = new QTemporaryFile();
        d->debugTempFile->setAutoRemove( false );
        d->debugTempFile->open();

        curl_easy_setopt(d->curl, CURLOPT_DEBUGFUNCTION, d->writeDebugToFile);
        curl_easy_setopt(d->curl, CURLOPT_DEBUGDATA, d->debugTempFile);
#endif

#endif
        const CURLcode result = curl_easy_perform(d->curl);
        if(CURLE_OK != result) {
            /* we failed */
            res = false;
            QFile tempFile( tempFileName );
            tempFile.remove();
            errorMessage = QString::fromUtf8( curl_easy_strerror( result ) );
            emit downloadError( errorMessage, result );
        } else {
            tempFile = tempFileName;
            res = true;
        }
        d->cleanup();
    }
    return res;
}

