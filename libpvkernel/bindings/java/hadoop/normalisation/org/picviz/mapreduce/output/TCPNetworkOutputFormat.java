package org.picviz.mapreduce.output;

import java.io.BufferedOutputStream;
import java.io.IOException;
import java.net.InetAddress;
import java.net.Socket;
import java.net.UnknownHostException;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.InvalidJobConfException;
import org.apache.hadoop.mapreduce.JobContext;
import org.apache.hadoop.mapreduce.JobStatus;
import org.apache.hadoop.mapreduce.JobStatus.State;
import org.apache.hadoop.mapreduce.OutputCommitter;
import org.apache.hadoop.mapreduce.RecordWriter;
import org.apache.hadoop.mapreduce.TaskAttemptContext;

public class TCPNetworkOutputFormat<K, V> extends NetworkOutputFormat<K, V> {

	protected static final String PORT = "mapreduce.output.tcpnetworkoutputformat.port";
	private OutputCommitter outputCommitter = null;

	protected class TCPRecordWriter extends RecordWriter<K, V> {
		protected Socket socket;
		protected BufferedOutputStream stream;

		public TCPRecordWriter(String host, int port, int id) throws UnknownHostException, IOException {
			socket = new Socket(host, port);
			stream = new BufferedOutputStream(socket.getOutputStream());
			// The first thing to write is the task ID
			stream.write((new Integer(id).toString() + "\n").getBytes("UTF-8"));
			// This is not a terminal task
			stream.write(new String("0\n").getBytes("UTF-8"));
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
			stream.flush();
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
	
	public static void sendLastFinishedTask(String host, int port) throws IOException {
		Socket socket_;
		try {
			socket_ = new Socket(host, port);
			BufferedOutputStream stream_ = new BufferedOutputStream(socket_.getOutputStream());
			// The task ID of the "final" task is the ID of the last task + 1
			stream_.write(new String("-1\n").getBytes("UTF-8"));
			stream_.write(new String("1\n").getBytes("UTF-8"));
			stream_.flush();
			socket_.close();
		} catch (UnknownHostException e) {
			System.out.println("In cleanupJob: exception 1");
			return;
		}
	}
	
	public static class TCPNetworkOutputCommitter extends OutputCommitter {
		public TCPNetworkOutputCommitter() {
			super();
		}
		@Override
		public void abortTask(TaskAttemptContext taskContext) throws IOException { System.out.println("In abortTasl"); }
		@Override
		public void abortJob(JobContext jobContext, JobStatus.State state) throws IOException { System.out.println("In abortJob"); cleanupJob(jobContext); }
		@Override
		public void cleanupJob(JobContext jobContext) throws IOException {
			// Send the "final" task to our host
			System.out.println("In cleanupJob");
			TCPNetworkOutputFormat.sendLastFinishedTask(TCPNetworkOutputFormat.getDestHost(jobContext), TCPNetworkOutputFormat.getDestPort(jobContext));
		}
		@Override
		public void commitJob(JobContext jobContext) throws IOException {
			System.out.println("In commitJob");
			cleanupJob(jobContext);
		}
		@Override
		public void commitTask(TaskAttemptContext taskContext) { System.out.println("In commitTask"); }
		@Override
		public boolean needsTaskCommit(TaskAttemptContext taskContext) {
			System.out.println("In needsTaskCommit");
			return true;
		}
		@Override
		public void setupJob(JobContext jobContext) { System.out.println("In setupJob"); }
		@Override
		public void setupTask(TaskAttemptContext taskContext) { System.out.println("In setupTask"); }
	}
	
	@Override
	public OutputCommitter getOutputCommitter(TaskAttemptContext task) throws IOException, InterruptedException  {
		return new TCPNetworkOutputCommitter();
	}

}

