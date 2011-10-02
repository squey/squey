#ifndef FILEDOWNLOADER_H
#define FILEDOWNLOADER_H

#include "logviewer_export.h"
#include "connectionsettings.h"

#include <QObject>
#include <QUrl>

class LOGVIEWER_EXPORT FileDownLoader : public QObject
{
    Q_OBJECT
public:
    explicit FileDownLoader(QObject * parent = 0);
    ~FileDownLoader();
    bool download( const QString &remoteFile, QString &tempFile, const ConnectionSettings& settings, const QString& hostName, QString& errorMessage, QUrl& url, bool& cancel );

signals:
    void downloadError( const QString&, int errorCode );

private:
    class FileDownLoaderPrivate;
    FileDownLoaderPrivate* d;
};

#endif /* FILEDOWNLOADER_H */

