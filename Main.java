import java.io.*;
import java.util.*;
import edu.stanford.nlp.ling.*;
import edu.stanford.nlp.neural.rnn.RNNCoreAnnotations;
import edu.stanford.nlp.pipeline.*;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations;
import edu.stanford.nlp.sentiment.SentimentCoreAnnotations;
import edu.stanford.nlp.trees.*;
import edu.stanford.nlp.trees.TreeCoreAnnotations.*;
import edu.stanford.nlp.util.*;
import org.apache.log4j.BasicConfigurator;

public class Main {

    public static void main(String[] args) throws IOException {
        // Configure logger
        BasicConfigurator.configure();

        // Initialize properties
        Properties props = new Properties();
        props.setProperty("annotators", "tokenize, ssplit, pos, lemma, ner, parse, sentiment");

        // Initialize pipeline
        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);

        // Initialize annotation
        Annotation annotation = new Annotation("This is a short sentence. And this is another. How will this parser deal with a more complex sentence structure? This is awesome. This is terrible");
        pipeline.annotate(annotation);

        for (CoreMap sentence : annotation.get(CoreAnnotations.SentencesAnnotation.class)) {
            PrintWriter out = new PrintWriter(System.out);
            out.println("\nSENTENSE: " + sentence);

            // Print word POSs
            for (CoreLabel token: sentence.get(CoreAnnotations.TokensAnnotation.class)) {
                out.println("Word: " +token.get(CoreAnnotations.TextAnnotation.class) + ", pos: " + token.get(CoreAnnotations.PartOfSpeechAnnotation.class));
            }

            // Print sentence graph and tree
            Tree tree = sentence.get(TreeAnnotation.class);
            out.println("Sentiment: " + RNNCoreAnnotations.getPredictedClass(sentence.get(SentimentCoreAnnotations.SentimentAnnotatedTree.class)) + " - " + sentence.get(SentimentCoreAnnotations.SentimentClass.class));
            SemanticGraph dependencies = sentence.get(SemanticGraphCoreAnnotations.CollapsedCCProcessedDependenciesAnnotation.class);
            out.println("Semantic Graph: " + dependencies);
            tree.pennPrint(out);
        }
    }
}