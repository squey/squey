package org.picviz.jni.PVRush;
import java.io.FileOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.BufferedOutputStream;

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
		// Auto-extract the library from the JAR archive
		try {
			String lib = ExtractFromJar("libpvrush_jni.so");
			System.load(lib);
			init();
		}
		catch (IOException e) {
			System.out.println("Unable to load the JNI library !");
		}
	}
	
	native static void init();
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
