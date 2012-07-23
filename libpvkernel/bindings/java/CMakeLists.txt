#
# \file CMakeLists.txt
#
# Copyright (C) Picviz Labs 2010-2012

if (JAVA_FOUND)
	if (JNI_FOUND)
		add_subdirectory(jni)
		add_subdirectory(hadoop)
	else (JNI_FOUND)
		MESSAGE(WARNING "JNI hasn't been found. Java bindings for libpvrush won't be available !")
	endif (JNI_FOUND)
else(JAVA_FOUND)
	MESSAGE(WARNING "Java hasn't been found. Java bindings for libpvrush won't be available !")
endif(JAVA_FOUND)
