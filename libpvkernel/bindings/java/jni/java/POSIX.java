// Use setenv/getenv from the libc to change current
// process' environnement.
// Inspired by http://quirkygba.blogspot.com/2009/11/setting-environment-variables-in-java.html

package org.picviz.common;

import com.sun.jna.Library;
import com.sun.jna.Native;

public class POSIX {
	public interface LibC extends Library {
		public int setenv(String name, String value, int overwrite);
		public int unsetenv(String name);
	}
	static public LibC libc = (LibC) Native.loadLibrary("c", LibC.class);
};

