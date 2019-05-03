/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVDISPLAYS_PVDISPLAYSIMPL_H
#define PVDISPLAYS_PVDISPLAYSIMPL_H

#include <QMetaType>

namespace PVDisplays
{

class PVDisplaysContainer;

class PVDisplaysImpl : public QObject
{
  private:
	PVDisplaysImpl()
	{
		// Load all plugins
		load_plugins();
	}

  public:
	static PVDisplaysImpl& get();

  private:
	void load_plugins();

  private:
	static PVDisplaysImpl* _instance;
};

} // namespace PVDisplays

#endif
