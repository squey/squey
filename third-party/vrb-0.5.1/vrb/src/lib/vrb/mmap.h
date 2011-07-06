#ifdef WIN32
#ifndef __WIN32_MMAP_H
#define __WIN32_MMAP_H

// from sys/mman.h
#define MAP_FAILED ((void*)-1)

void *mmap(void *ptr, long size, long prot, long type, long handle, long arg);
long munmap(void *ptr, long size);

#endif //__WIN32_MMAP_H
#endif //WIN32
