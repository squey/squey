/**
 * \file PVUtils.cpp
 *
 * Copyright (C) Picviz Labs 2011-2012
 */

#include <pvkernel/core/PVUtils.h>

#ifdef WIN32
	#include <windows.h>
#else
	#include <X11/Xlib.h>
	#include <X11/XKBlib.h>
#endif


// http://www.qtforum.org/article/32572/how-to-determine-if-capslock-is-on-crossplatform.html
bool PVCore::PVUtils::isCapsLockActivated()
{
	// platform dependent method of determining if CAPS LOCK is on
#ifdef WIN32 // MS Windows version
	return GetKeyState(VK_CAPITAL) == 1;
#else // X11 version (Linux/Unix/Mac OS X/etc...)
	Display * d = XOpenDisplay((char*)0);
	bool caps_state = false;
	if (d)
		{
			unsigned n;
			XkbGetIndicatorState(d, XkbUseCoreKbd, &n);
			caps_state = (n & 0x01) == 1;
		}
	return caps_state;
#endif
}

