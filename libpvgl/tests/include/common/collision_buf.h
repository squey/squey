#ifndef tests_collision_buffer_h
#define tests_collision_buffer_h

typedef int* DECLARE_ALIGN(16) CollisionBuffer;

CollisionBuffer allocate_CB(void);
void free_CB(CollisionBuffer b);

#define PIXELS_CB 512
#define CB_INT_OFFSET(bit) ((bit>>5))
#define CB_BITN(bit) ((bit&31))

#define NB_INT_CB (PIXELS_CB*PIXELS_CB/32)
#define SIZE_CB (NB_INT_CB*sizeof(int))

#endif
