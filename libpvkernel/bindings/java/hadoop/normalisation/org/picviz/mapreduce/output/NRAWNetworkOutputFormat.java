package org.picviz.mapreduce.output;

import java.io.IOException;
import java.net.UnknownHostException;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.RecordWriter;
import org.apache.hadoop.mapreduce.TaskAttemptContext;

public class NRAWNetworkOutputFormat<K, V> extends TCPNetworkOutputFormat<K, V> {
	protected class NRAWRecordWriter extends TCPRecordWriter {
		public NRAWRecordWriter(String host, int port, int id)
				throws UnknownHostException, IOException {
			super(host, port, id);
		}

		public byte[] longToBytes(long v) {
		    byte[] writeBuffer = new byte[ 8 ];

		    writeBuffer[0] = (byte)(v >>> 56);
		    writeBuffer[1] = (byte)(v >>> 48);
		    writeBuffer[2] = (byte)(v >>> 40);
		    writeBuffer[3] = (byte)(v >>> 32);
		    writeBuffer[4] = (byte)(v >>> 24);
		    writeBuffer[5] = (byte)(v >>> 16);
		    writeBuffer[6] = (byte)(v >>>  8);
		    writeBuffer[7] = (byte)(v >>>  0);

		    return writeBuffer;
		}
		
		public byte[] intToBytes(long v) {
		    byte[] writeBuffer = new byte[ 4 ];

		    writeBuffer[0] = (byte)(v >>> 24);
		    writeBuffer[1] = (byte)(v >>> 16);
		    writeBuffer[2] = (byte)(v >>>  8);
		    writeBuffer[3] = (byte)(v >>>  0);

		    return writeBuffer;
		}

		public void writeLong(LongWritable l) throws IOException
		{
			stream.write(longToBytes(l.get()));
		}
		
		public void writeInt(int l) throws IOException
		{
			stream.write(intToBytes(l));
		}
		
		public void writeString(String s) throws IOException
		{
			byte[] sUtf8 = s.getBytes("UTF-8");
			writeInt(sUtf8.length);
			stream.write(sUtf8);
		}
		
		
		public synchronized void write(K key, V value) throws IOException
		{
			if (value instanceof String[] && key instanceof LongWritable) {
				String[] fields = (String[]) value;
				writeLong((LongWritable) key);
				int lengthElt = 4;
				for (int i = 0; i < fields.length; i++) {
					lengthElt += fields[i].length();
				}
				writeInt(lengthElt);
				for (int i = 0; i < fields.length; i++) {
					writeString(fields[i]);
				}
			}
		}

	}

	public RecordWriter<K, V> getRecordWriter(TaskAttemptContext job) throws IOException, InterruptedException {
		int id = job.getTaskAttemptID().getTaskID().getId();
		return new NRAWRecordWriter(getDestHost(job), getDestPort(job), id);
	}
}
