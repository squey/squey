/**
 * \file PVRushJNI.java
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

package org.picviz.jni.PVRush;
import java.io.FileOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.BufferedOutputStream;

import org.apache.commons.vfs.*;

//import org.picviz.common.POSIX;

public class PVRushJNI {
	
	public static String ExtractFromJar(String name) throws IOException {
		File directory = new File(System.getProperty("java.io.tmpdir"));
		if(!directory.exists())
			directory.mkdirs();
		File libFile = new File(directory, name);
		if(libFile.exists())
			libFile.delete();
		InputStream in = PVRushJNI.class.getResourceAsStream("/" + name);
		BufferedOutputStream out = new BufferedOutputStream(new FileOutputStream(libFile));
		byte buffer[] = new byte[1024];
		int len;
		for(int sum = 0; (len = in.read(buffer)) > 0; sum += len)
			out.write(buffer, 0, len);
		in.close();
		out.close();
		return libFile.getAbsolutePath();
	}

	static {
		// Auto-extract the libraries from the JAR archive
		try {
			String libs_path_tar = ExtractFromJar("pvkernel_libs.tar");
			// Extract all the libs and load pvkernel
			String tmp_dir_path = System.getProperty("java.io.tmpdir") + "/pvkernel-jni-libs/";
			File tmp_dir = new File(tmp_dir_path);
			tmp_dir.mkdirs();
			FileSystemManager fsManager = VFS.getManager();
			FileObject lib_tar = fsManager.resolveFile("tar:" + libs_path_tar);
			FileObject[] libs = lib_tar.getChildren();
			for (int i = 0; i < libs.length; i++) {
				FileObject lfo = libs[i];
				InputStream lib_content = lfo.getContent().getInputStream();
				String lib_name = lfo.getName().getBaseName();
				File libFile = new File(tmp_dir_path, lib_name);
				BufferedOutputStream out = new BufferedOutputStream(new FileOutputStream(libFile));
				byte[] buffer = new byte[1024];
				int len;
				for (int sum = 0; (len = lib_content.read(buffer)) > 0; sum += len) {
					out.write(buffer, 0, len);
				}
				lib_content.close();
				out.close();
			}

			// Set LD_LIBRARY_PATH to tmp_dir_path
			// This does *not* work, even if setting LD_LIBRARY_PATH by hand
			// before launching java *does* work...
			// We load pvkernel by hand waiting for better :s
			/* ...design dead with no option...
			String ldlibp = POSIX.libc.getenv("LD_LIBRARY_PATH");
			System.out.println("LD_LIBRARY_PATH was " + ldlibp);
			ldlibp += ":" + tmp_dir_path;
			System.out.println("Setting LD_LIBRARY_PATH to " + ldlibp + "...");
			POSIX.libc.setenv("LD_LIBRARY_PATH", ldlibp, 1);
			System.out.println("LD_LIBRARY_PATH is now " + POSIX.libc.getenv("LD_LIBRARY_PATH"));
			*/

			System.load(tmp_dir_path + "libpvkernel.so.1");
			System.out.println("pvkernel loaded");
			System.load(tmp_dir_path + "libpvrush_jni.so");
			init(tmp_dir_path);
		}
		catch (IOException e) {
			System.out.println("Unable to load the JNI library !");
			System.out.println(e.toString());
			e.printStackTrace();
		}
	}
	
	native private static void init(String libs_dir);
	native public void init_with_format(String path_format);
	native public String[] process_elt(String elt);
	
	public static void main(String[] args) {
		PVRushJNI rush = new PVRushJNI();
		String path = new String("/home/aguinet/pv/apache.access.noise.log.format");
		System.out.println(path);
		rush.init_with_format(path);
		String[] arr = rush.process_elt(new String("kimé.geekou.info:80 208.80.194.33 - - [26/Jul/2011:03:11:51 +0200] \"GET / HTTP/1.0\" 403 396 \"-\" \"éMozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; FunWebProducts; Wanadoo 6.2; .NET CLR 1.1.4322; HbTools 4.8.4)\""));
		for (int i = 0; i < arr.length; i++) {
			System.out.println(arr[i]);
		}
	}
}
