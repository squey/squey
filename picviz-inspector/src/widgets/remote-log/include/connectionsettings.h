#ifndef CONNECTIONSETTINGS_H
#define CONNECTIONSETTINGS_H

#include <QString>

enum Protocol {
    Local = 0,
    Http,
    Https,
    Ftp,
    Ftps,
    Scp,
    SFtp
};

struct ConnectionSettings
{
    ConnectionSettings()
        : port( 0 ), protocol( Local ), ignoreSslError( true )
    {
    }
    QString sshKeyFile;
    QString password;
    QString login;
    QString certificatFile;
    int port;
    Protocol protocol;
    bool ignoreSslError;
};


struct RegisteredFile {
    QString remoteFile;
    QString localFile;
    ConnectionSettings settings;
};

struct MachineConfig {
    MachineConfig( const QString& _name = QString(), const QString& _hostname = QString() )
        : name( _name ), hostname( _hostname )
    {
    }
    QString name;
    QString hostname;
};

inline bool operator<(const MachineConfig &e1, const MachineConfig &e2)
{
    if (e1.name != e2.name )
        return e1.name < e2.name;
    return e1.hostname < e2.hostname;
}




#endif /* CONNECTIONSETTINGS_H */

