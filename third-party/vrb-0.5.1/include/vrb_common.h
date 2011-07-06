//-----------------------------------------------------------------------------
// Copyright © 2006 - Philip Howard - All rights reserved
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation; either version 2
// of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
//-----------------------------------------------------------------------------
// package	vrb
// homepage	http://vrb.slashusr.org/
//-----------------------------------------------------------------------------
// author	Philip Howard
// email	vrb at ipal dot org
// homepage	http://phil.ipal.org/
//-----------------------------------------------------------------------------
// This file is best viewed using a fixed spaced font such as Courier
// and in a display at least 120 columns wide.
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// file		common.h
//
// purpose	Common header included by all library-only headers during
//		library compilation.
//-----------------------------------------------------------------------------
#ifndef __VRB_COMMON_H__
#define __VRB_COMMON_H__

//-----------------------------------------------------------------------------
// configuration
//
// Define symbols used for extracting header code so they will be ignored
// during compilation.
//
// prefix include macro typedef proto inline suffix
//-----------------------------------------------------------------------------
#ifndef __PREFIX_BEGIN__
#define __PREFIX_BEGIN__
#endif
#ifndef __PREFIX_END__
#define __PREFIX_END__
#endif

#ifndef __INCLUDE_BEGIN__
#define __INCLUDE_BEGIN__
#endif
#ifndef __INCLUDE_END__
#define __INCLUDE_END__
#endif

#ifndef __CONFIG_BEGIN__
#define __CONFIG_BEGIN__
#endif
#ifndef __CONFIG_END__
#define __CONFIG_END__
#endif

#ifndef __DEFINE_BEGIN__
#define __DEFINE_BEGIN__
#endif
#ifndef __DEFINE_END__
#define __DEFINE_END__
#endif

#ifndef __FMACRO_BEGIN__
#define __FMACRO_BEGIN__
#endif
#ifndef __FMACRO_END__
#define __FMACRO_END__
#endif

#ifndef __PROTO_BEGIN__
#define __PROTO_BEGIN__
#endif
#ifndef __PROTO_END__
#define __PROTO_END__
#endif

#ifndef __INLINE_BEGIN__
#define __INLINE_BEGIN__
#endif
#ifndef __INLINE_END__
#define __INLINE_END__
#endif

#ifndef __ALIAS_BEGIN__
#define __ALIAS_BEGIN__
#endif
#ifndef __ALIAS_END__
#define __ALIAS_END__
#endif

#ifndef __SUFFIX_BEGIN__
#define __SUFFIX_BEGIN__
#endif
#ifndef __SUFFIX_END__
#define __SUFFIX_END__
#endif

//-----------------------------------------------------------------------------
// end header
//-----------------------------------------------------------------------------
#endif /* __VRB_COMMON_H__ */
