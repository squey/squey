/**
 * \file process.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pthread.h>

#include <pvkernel/core/pvbuffer.h>
#include <pvkernel/core/pvprocess.h>

#include <iostream>
#include <unistd.h>

using namespace PVCore;

PVBuffer *buf = new PVBuffer();
PVProcess1 *p1 = new PVProcess1();
PVProcess2 *p2 = new PVProcess2();


void *mythread1(void *threadid)
{
	p1->process(1000000);

	pthread_exit(NULL);
}



void *mythread2(void *threadid)
{
	p2->process(1000000);

	pthread_exit(NULL);
}



int main(void)
{
	pthread_t threads[12];
	int rc;
	long t;

	// Initialization
	p1->output_buffer = buf;
	p1->output_pvprocess = p2;
	p2->input_buffer = buf;
	p2->input_pvprocess = p1;


	
	t = 0;
	rc = pthread_create(&threads[t], NULL, mythread1, (void *)t);
	t = 1;
	rc = pthread_create(&threads[t], NULL, mythread2, (void *)t);

	pthread_exit(NULL);

	return 0;
}
