/**
 * \file file.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/rush/file.h>

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
#include "test-env.h"

  PVRush::File file("test-files/file_ending.utf16.gz");
  QByteArray qba;

  printf("code name=%s\n", file.codec->name().data());
//   printf("is compressed=%d\n", file.is_compressed);


  file.Uncompress(QString("test-files/file_ending.utf16.gz"),QString("outfile"));
//   qba = file.file.read(12);
//   printf("some data='%s'\n", qba.data());


  return 0;
}

