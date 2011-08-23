package org.picviz.mapreduce.output;

import java.io.BufferedOutputStream;
import java.io.IOException;
import java.net.InetAddress;
import java.net.Socket;
import java.net.UnknownHostException;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.InvalidJobConfException;
import org.apache.hadoop.mapreduce.JobContext;
import org.apache.hadoop.mapreduce.RecordWriter;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.TaskAttemptID;

public class TCPNetworkOutputFormat<K, V> extends NetworkOutputFormat<K, V> {

	protected static final String PORT = "mapreduce.output.tcpnetworkoutputformat.port";

	protected class TCPRecordWriter extends RecordWriter<K, V> {
		protected Socket socket;
		protected BufferedOutputStream stream;

		public TCPRecordWriter(String host, int port, int id) throws UnknownHostException, IOException {
			socket = new Socket(host, port);
			stream = new BufferedOutputStream(socket.getOutputStream());
			// The first thing to write is the task ID
			stream.write(new Integer(id).toString().getBytes("UTF-8"));
		}

		protected void writeObject(Object o) throws IOException {
			if (o instanceof Text) {
				Text to = (Text) o;
				stream.write(to.getBytes(), 0, to.getLength());
			} else {
				stream.write(o.toString().getBytes("UTF-8"));
			}
		}

		public synchronized void write(K key, V value) throws IOException
		{
			writeObject(key);
			writeObject(value);
		}

		@Override
		public void close(TaskAttemptContext arg0) throws IOException,
				InterruptedException {
			socket.close();
			
		}
	}

	public RecordWriter<K, V> getRecordWriter(TaskAttemptContext job) throws IOException, InterruptedException {
		int id = job.getTaskAttemptID().getTaskID().getId();
		return new TCPRecordWriter(getDestHost(job), getDestPort(job), id);
	}

	public void checkOutputSpecs(JobContext job) throws UnknownHostException, InvalidJobConfException {
		String host = getDestHost(job);
		if (host == null) {
			throw new InvalidJobConfException("Port is invalid.");
		}
		@SuppressWarnings("unused")
		InetAddress addr = InetAddress.getByName(host);
	}

	public static void setDestPort(JobContext job, int port) {
		job.getConfiguration().setInt(TCPNetworkOutputFormat.PORT, port);
	}

	public static int getDestPort(JobContext job) {
		int port = job.getConfiguration().getInt(TCPNetworkOutputFormat.PORT, 0);
		return port;
	}

}

