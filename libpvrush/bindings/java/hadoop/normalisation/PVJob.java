import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class PVJob {
	public static void main(String[] args) throws Exception {
		Configuration cfg = new Configuration();
		cfg.setStrings("mapreduce.pvjob.format_path", args[2]);

		Job job = new Job(cfg);
		job.setJarByClass(grep.class);

		job.setJobName("pvjob_normalisation");

		FileInputFormat.addInputPath(job, new Path(args[0]));
		FileOutputFormat.setOutputPath(job, new Path(args[1]));

		job.setMapperClass(PVMapper.class);
		job.setCombinerClass(PVReduce.class);
		job.setReducerClass(PVReduce.class);

		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(LongWritable.class);

		job.waitForCompletion(true);
	}
}

