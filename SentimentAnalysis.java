import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.neural.rnn.RNNCoreAnnotations;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.sentiment.SentimentCoreAnnotations;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.util.CoreMap;
import org.apache.log4j.BasicConfigurator;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;

public class SentimentAnalysis {

    private static List<Integer> getSentiment(String path) throws IOException {
        // Configure logger
        BasicConfigurator.configure();

        // Get file paths
        List<String> files = new ArrayList<>();
        Files.list(Paths.get(path)).forEach(f -> files.add(f.toString()));

        // Read text files into list of strings
        List<String> reviews = new ArrayList<>();
        files.forEach(f -> {
            try {
                reviews.add(new String(Files.readAllBytes(Paths.get(f))));
            } catch (IOException e) {
                e.printStackTrace();
            }
        });

        // Initialize properties and pipeline
        Properties props = new Properties();
        props.setProperty("annotators", "tokenize, ssplit, pos, parse, sentiment");
        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);

        List<Integer> review_sentiments = new ArrayList<>();
        for (int i = 0; i < reviews.size(); i++) {
            // Initialize annotation
            Annotation annotation = new Annotation(reviews.get(i));
            pipeline.annotate(annotation);

            // Get review sentiment by weighing each sentence sentiment by number of tokens in sentence
            int sentiment = 0;
            for (CoreMap sentence : annotation.get(CoreAnnotations.SentencesAnnotation.class)) {
                Tree sentiment_tree = sentence.get(SentimentCoreAnnotations.SentimentAnnotatedTree.class);
                int sentence_sentiment = RNNCoreAnnotations.getPredictedClass(sentiment_tree) - 2;
                int num_tokens = sentence.get(CoreAnnotations.TokensAnnotation.class).size();
                sentiment += sentence_sentiment * num_tokens;
            }
            review_sentiments.add(sentiment);

            // Print inference progress
            if (i % 1000 == 0) {
                System.out.println(i + " reviews processed");
            }
        }

        return review_sentiments;
    }

    public static void main(String[] args) throws IOException {
        List<Integer> posSentiment = getSentiment("src/main/data/aclImdb/test/pos/");
        List<Integer> negSentiment = getSentiment("src/main/data/aclImdb/test/neg/");

        double posAverage = posSentiment.stream().mapToInt(Integer::intValue).average().getAsDouble();
        double negAverage = negSentiment.stream().mapToInt(Integer::intValue).average().getAsDouble();
        double posAccuracy = posSentiment.stream().filter(i -> i > 0).count() / (double) posSentiment.size();
        double negAccuracy = negSentiment.stream().filter(i -> i < 0).count() / (double) negSentiment.size();

        System.out.println("\nStatistics:");
        System.out.println("Average weighted sentiment for positive reviews: " + posAverage);        // -44.51
        System.out.println("Average weighted sentiment for negative reviews: " + negAverage);        // -198.69
        System.out.println("Accuracy for positive test set reviews: " + posAccuracy);                // 43.18%
        System.out.println("Accuracy for negative test set reviews: " + negAccuracy);                // 96.74%
        System.out.println("Accuracy for all test set reviews: " + (posAccuracy + negAccuracy) / 2); // 69.96%
    }
}