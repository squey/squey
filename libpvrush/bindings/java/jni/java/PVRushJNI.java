public class PVRushJNI {
	static {
		System.loadLibrary("pvrush_jni");
	}
	
	native void init_with_format(String path_format);
	native String[] process_elt(String elt);
	
	public static void main(String[] args) {
		PVRushJNI rush = new PVRushJNI();
		String path = new String("/home/aguinet/pv/apache.access.noise.log.format");
		System.out.println(path);
		rush.init_with_format(path);
		String[] arr = rush.process_elt(new String("kim.geekou.info:80 208.80.194.33 - - [26/Jul/2011:03:11:51 +0200] \"GET / HTTP/1.0\" 403 396 \"-\" \"Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; FunWebProducts; Wanadoo 6.2; .NET CLR 1.1.4322; HbTools 4.8.4)\""));
		for (int i = 0; i < arr.length; i++) {
			System.out.println(arr[i]);
		}
	}
}
