/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef CONNECTIONSETTINGS_H
#define CONNECTIONSETTINGS_H

#include <QString>

enum Protocol {
    Http = 0,
    Https,
    Ftp,
    Ftps,
    Scp,
    SFtp,
    Local
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

