//-----------------------------------------------------------------------------
// Copyright © 2002 - Philip Howard - All rights reserved
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	 See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307	 USA
//-----------------------------------------------------------------------------
// package	vrb
// homepage	http://phil.ipal.org/freeware/vrb/
//-----------------------------------------------------------------------------
// author	Philip Howard
// email	vrb at ipal dot org
// homepage	http://phil.ipal.org/
//-----------------------------------------------------------------------------
// This file is best viewed using a fixed spaced font such as Courier
// and in a display at least 120 columns wide.
//-----------------------------------------------------------------------------
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef WIN32
#include "mmap.h"
#else
#include <sys/mman.h>
#endif
#include <sys/shm.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "vrb_lib.h"

//-----------------------------------------------------------------------------
// MAP_ANONYMOUS probably is defined but if not, maybe MAP_ANON is.
//-----------------------------------------------------------------------------
#ifndef MAP_ANONYMOUS
#define MAP_ANONYMOUS MAP_ANON
#endif

__FMACRO_BEGIN__
//-----------------------------------------------------------------------------
// macro	vrb_new
//
// purpose	Make a new instance of a vrb, returning the pointer to the
//		vrb structure.
//
//		If no name is given, use SysV shared memory for the shared
//		memory object, and swap for backing store.
//
//		If a file name pattern is given, use POSIX memory mapping
//		into a temporary file created from that file name pattern
//		for the shared memory object, using file for backing store.
//
// arguments	1 (size_t) minimum buffer space requested
//		2 (const char *) name (temp pattern or actual) to memory map
//
// note		The name argument is a temporary filename pattern used by
//		mkstemp() to create a unique temporary file if it ends with
//		the string "XXXXXX" as required by mkstemp(), or is used as
//		an actual filename if it does not end with that string.
//
// returns	(vrb_p) pointer to vrb
//		(vrb_p) NULL if error (see errno)
//-----------------------------------------------------------------------------
#define vrb_new(s,n) (vrb_init_opt(((vrb_p)(NULL)),(s),(n),0))
__FMACRO_END__

__FMACRO_BEGIN__
//-----------------------------------------------------------------------------
// macro	vrb_new_opt
//
// purpose	Make a new instance of a vrb, returning the pointer to the
//		vrb structure.
//
//		This version includes an option flags argument.
//
//		If no name is given, use SysV shared memory for the shared
//		memory object, and swap for backing store.
//
//		If a file name pattern is given, use POSIX memory mapping
//		into a temporary file created from that file name pattern
//		for the shared memory object, using file for backing store.
//
// arguments	1 (size_t) minimum buffer space requested
//		2 (const char *) name (temp pattern or actual) to memory map
//		3 (int) option flags
//
// options	VRB_NOGUARD	Do not add guard pages around the buffer
//
//		VRB_ENVGUARD	Override VRB_NOGUARD if the environment
//				variable VRBGUARD is defined other than
//				the string "0"
//
//		VRB_NOMKSTEMP	Do not use mkstemp() to make a temporary
//				file name from the given pattern.  Use it
//				directly as the file name.
//
// note		The name argument is a temporary filename pattern used by
//		mkstemp() to create a unique temporary file if it ends with
//		the string "XXXXXX" as required by mkstemp(), or is used as
//		an actual filename if it does not end with that string.
//
// returns	(vrb_p) pointer to vrb
//		(vrb_p) NULL if error (see errno)
//-----------------------------------------------------------------------------
#define vrb_new_opt(s,n,o) (vrb_init_opt(((vrb_p)(NULL)),(s),(n),(o)))
__FMACRO_END__

__FMACRO_BEGIN__
//-----------------------------------------------------------------------------
// macro	vrb_init
//
// purpose	Initialize a vrb with an empty space of at least the size
//		requested.  The actual size will be rounded up to meet system
//		requirements, if needed.  Additional address space will be
//		used for guard pages added to prevent buffer overflow errors.
//		
//		If no vrb is given, create a new one.
//
//		If no name is given, use SysV shared memory for the shared
//		memory object, and swap for backing store.
//
//		If a file name pattern is given, use POSIX memory mapping
//		into a temporary file created from that file name pattern
//		for the shared memory object, using file for backing store.
//
// arguments	1 (vrb_p) NULL or an existing vrb to initialize
//		2 (size_t) buffer size, which will be rounded up as needed
//		3 (const char *) name (temp pattern or actual) to memory map
//
// note		The name argument is a temporary filename pattern used by
//		mkstemp() to create a unique temporary file if it ends with
//		the string "XXXXXX" as required by mkstemp(), or is used as
//		an actual filename if it does not end with that string.
//
// returns	(vrb_p) pointer to initialized vrb
//		(vrb_p) NULL if there was an error
//-----------------------------------------------------------------------------
#define vrb_init(v,s,n) (vrb_init_opt((v),(s),(n),0))
__FMACRO_END__

__PROTO_BEGIN__
//-----------------------------------------------------------------------------
// function	vrb_init_opt
//
// purpose	Initialize a vrb with an empty space of at least the size
//		requested.  The actual size will be rounded up to meet system
//		requirements, if needed.  Additional address space will be
//		used for guard pages added to prevent buffer overflow errors.
//
//		This version includes an option flags argument.
//		
//		If no vrb is given, create a new one.
//
//		If no name is given, use SysV shared memory for the shared
//		memory object, and swap for backing store.
//
//		If a file name pattern is given, use POSIX memory mapping
//		into a temporary file created from that file name pattern
//		for the shared memory object, using file for backing store.
//
// arguments	1 (vrb_p) NULL or an existing vrb to initialize
//		2 (size_t) buffer size, which will be rounded up as needed
//		3 (const char *) name (temp pattern or actual) to memory map
//		4 (int) option flags
//
// options	VRB_NOGUARD	Do not add guard pages around the buffer
//
//		VRB_ENVGUARD	Override VRB_NOGUARD if the environment
//				variable VRBGUARD is defined other than
//				the string "0"
//
//		VRB_NOMKSTEMP	Do not use mkstemp() to make a temporary
//				file name from the given pattern.  Use it
//				directly as the file name.
//
// note		The name argument is a temporary filename pattern used by
//		mkstemp() to create a unique temporary file if it ends with
//		the string "XXXXXX" as required by mkstemp(), or is used as
//		an actual filename if it does not end with that string.
//
// returns	(vrb_p) pointer to initialized vrb
//		(vrb_p) NULL if there was an error
//-----------------------------------------------------------------------------
vrb_p
vrb_init_opt (
    vrb_p		arg_vrb
    ,
    size_t		arg_size
    ,
    const char *	arg_name
    ,
    int			arg_option
    )
    __PROTO_END__
{
    char *	mem_ptr		;
    char *	lower_ptr	;
    char *	upper_ptr	;

    vrb_p	vrb_ptr		;

    size_t	guard		;
    size_t	page_size	;
    size_t	req_size	;

    int		fd		;
    int		shm_id		;
    int		save_errno	;

    //--------------------------------------------------------------
    // Initialize cleanup flags to indicate nothing yet to clean up.
    //--------------------------------------------------------------
    fd			= -1;
    shm_id		= -1;

    //-----------------------------------------------
    // Get the system page size one way or the other.
    //-----------------------------------------------
#ifdef _SC_PAGESIZE
    page_size = sysconf( _SC_PAGESIZE );
#else
    page_size = getpagesize();
#endif

    //-------------------------------------------------
    // Decide if guard pages are to be used based on
    // either request by the caller or the environment.
    //-------------------------------------------------
    guard = ( arg_option & VRB_NOGUARD &&
	      ( ! ( arg_option & VRB_ENVGUARD ) ||
		! ( mem_ptr = getenv( "VRBGUARD" ) ) ||
		( mem_ptr[0] == '0' && mem_ptr[1] == 0 ) ) )
	? 0 : page_size;

    //----------------------------------------------------------------------
    // Make sure that the requested size is small enough so that arithmetic
    // operations applied to it of rounding up to a multiple of a page size,
    // adding a guard page size, and doubling that size, will not exceed the
    // largest value that can be represented in a signed type.  This will
    // ensure that arithmetic does not over flow and that correct values
    // are passed in system calls.  Any further size limitations imposed by
    // the operating system can then be correctly determined and handled.
    //----------------------------------------------------------------------
    if ( arg_size > ( ( (~(size_t)0) >> 2 ) - page_size - guard - guard ) ) {
        errno = EINVAL;
	goto error_return;
    }

    //---------------------------------------------------------------
    // Round the requested size up to an exact multiple of the system
    // page size, making sure there is at least one whole page.
    //---------------------------------------------------------------
    if ( ! arg_size ) arg_size = 1;
    req_size = arg_size + ( page_size - 1 );
    req_size -= req_size & ( page_size - 1 );

    //-----------------------------------------------------
    // If no VRB structure was provided, then allocate one.
    //-----------------------------------------------------
    if ( ! ( vrb_ptr = arg_vrb ? arg_vrb : malloc( sizeof (struct vrb) ) ) )
	goto error_malloc;
    vrb_ptr->buf_size		= req_size;
    vrb_ptr->flags		= 0;
    
    //------------------------------------------------------------------------
    // In order to place two virtual memory segments at adjacent locations, do
    // an anonymous mmap() call to find one contiguous space twice as large.
    // This allocation also includes the guard pages if requested.
    //------------------------------------------------------------------------
    mem_ptr = (char *) mmap( 0, // no fixed ptr
                             req_size + req_size + guard + guard,
                             PROT_NONE,
                             MAP_ANONYMOUS | MAP_PRIVATE,
                             -1, // no fd
                             0 );
    if ( mem_ptr == (char *) MAP_FAILED ) goto error_mmap_find;
    vrb_ptr->mem_ptr = mem_ptr;

    vrb_ptr->lower_ptr = vrb_ptr->first_ptr = vrb_ptr->last_ptr =
    lower_ptr = mem_ptr + guard;

    vrb_ptr->upper_ptr =
    upper_ptr = lower_ptr + req_size;

    //----------------------------------------------------------
    // If no filename is given, then use SYSV shared memory.
    // This uses swapping space as backing store, which is what
    // is expected of non-file-space memory usage.  An anonymous
    // mmap() cannot be used since there is no way to map the
    // same anonymous object into two address space locations.
    //----------------------------------------------------------
    if ( ! arg_name ) {

        //-- Unmap the portion to be used leaving requested guard pages at the ends.
        if ( 0 > munmap( lower_ptr, req_size + req_size ) ) goto error_munmap;

#ifdef WIN32
		HANDLE hMap;
		hMap = CreateFileMapping(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, req_size, "vrb");
		if (hMap == NULL)
			goto error_shmget;

		//TODO: finish this
#endif

        //-- Create a private shared memory segment.
        if ( 0 > ( shm_id = shmget( IPC_PRIVATE, req_size, IPC_CREAT | 0700 ) ) ) goto error_shmget;

        //-- Attach the shared segment at both lower and upper locations.
        if ( lower_ptr != shmat( shm_id, lower_ptr, 0 ) ) goto error_shmat_lower;
        if ( upper_ptr != shmat( shm_id, upper_ptr, 0 ) ) goto error_shmat_upper;

        //-- Mark the segment to be deleted after last detach.
        if ( 0 > shmctl( shm_id, IPC_RMID, NULL ) ) goto error_shmctl;

	vrb_set_shmat( vrb_ptr );
    }

    //------------------------------------------------------------
    // Else use POSIX memory mapping with the specified file name.
    //------------------------------------------------------------
    else {
	{
	    char * tempname;
	    //-- Try to open the file name as an mkstemp name pattern.
	    if ( ! ( tempname = alloca( strlen( arg_name ) + 1 ) ) ) goto error_alloca;
	    strcpy( tempname, arg_name );
	    if ( 0 <= ( fd = mkstemp( tempname ) ) ) arg_name = tempname;

	    //-- If mkstemp failed, try the original name as is.
	    else if ( 0 > ( fd = open( arg_name, O_RDWR | O_EXCL, 0600 ) ) ) goto error_open;

	    //-- Unlink whichever file was opened.
	    if ( 0 > unlink( arg_name ) ) goto error_unlink;
	}

	{
	    struct stat stat_buf;
	    //-- Expand file with ftruncate or if that fails, with lseek.
	    if ( 0 > ftruncate( fd, req_size ) ||
		 0 > fstat( fd, & stat_buf ) ||
		 stat_buf.st_size != req_size ) {
		if ( 0 > (int) lseek( fd, req_size - 1, SEEK_CUR ) ||
		     0 > write( fd, "", 1 ) ||
		     0 > fstat( fd, & stat_buf ) ||
		     stat_buf.st_size != req_size ) goto error_expand;
	    }
	}

        //-- Map the file over both the lower and upper locations.
        if ( mmap( lower_ptr,
                   req_size,
                   PROT_READ | PROT_WRITE,
                   MAP_FIXED | MAP_SHARED,
                   fd,
                   0 ) != lower_ptr ) goto error_mmap_lower;
        if ( mmap( upper_ptr,
                   req_size,
                   PROT_READ | PROT_WRITE,
                   MAP_FIXED | MAP_SHARED,
                   fd,
                   0 ) != upper_ptr ) goto error_mmap_upper;

        //-- Close the (unlinked) file now that its space is mapped.
        close( fd );

	vrb_set_mmap( vrb_ptr );
    }

    //--------------------------------------------
    // Fill in and return the vrb header struct.
    //--------------------------------------------
    return vrb_ptr;

    //--------------------------------------------------------------
    // Do all the failure clean up here, then return an error value.
    //--------------------------------------------------------------

    //-- Cleanup track for SYSV shared memory:
 error_shmctl:
    save_errno = errno;
    shmdt( upper_ptr );
    errno = save_errno;

 error_shmat_upper:
    save_errno = errno;
    shmdt( lower_ptr );
    errno = save_errno;

 error_shmat_lower:
    save_errno = errno;
    shmctl( shm_id, IPC_RMID, NULL );
    errno = save_errno;
    goto error_common;

    //-- Cleanup track for POSIX memory mapping:
 error_mmap_upper:
    save_errno = errno;
    munmap( lower_ptr, req_size );
    errno = save_errno;

 error_mmap_lower:
 error_expand:
 error_unlink:
    save_errno = errno;
    close( fd );
    errno = save_errno;

    //-- Common cleanup track:
 error_open:
 error_alloca:
 error_shmget:
 error_munmap:

 error_common:
    if ( guard ) {
        munmap( mem_ptr, page_size );
        munmap( mem_ptr + req_size + req_size, page_size );
    }

 error_mmap_find:
    memset( vrb_ptr, 0, sizeof (struct vrb) );
    if ( ! arg_vrb ) free( vrb_ptr );

 error_malloc:
 error_return:
    return NULL;

}
