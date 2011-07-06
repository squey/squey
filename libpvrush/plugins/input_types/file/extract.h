#ifndef INPUTTYPEFILE_EXTRACT_H
#define INPUTTYPEFILE_EXTRACT_H

#include <QStringList>
#include <QString>

bool is_archive(QString const& path);
bool extract_archive(QString const& path, QString const& dir_dest, QStringList &extracted_files);

#endif
