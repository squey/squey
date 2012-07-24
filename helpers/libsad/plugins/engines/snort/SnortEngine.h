/**
 * \file SnortEngine.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef SNORTENGINE_H
#define SNORTENGINE_H

#include <sad/general.h>

#include <QString>
#include <QStringList>

namespace Sad {

class SnortEngine
{
public:
	SnortEngine();
	virtual ~SnortEngine();
public:
	QString name() const;

protected:
	bool load_rules(QString const& rulespath) const;
	
	CLASS_REGISTRABLE_NOCOPY(SnortEngine)
};

}

#endif
