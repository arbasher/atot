using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using System.Diagnostics.Contracts;

namespace Modeling
{
    class Program
    {
        private static void Main(string[] args)
        {
            Console.WriteLine("Please choose: Topic Modeling (M) or any key to exit:");
            string choice = Console.ReadLine().ToLower();

            while (choice == "m")
            {
                Console.WriteLine("Please choose: D (LDA), A (AT), L (LDA-TOT) or T (A-TOT) or B (CT)");
                string modeltype = Console.ReadLine().ToLower();

                Console.WriteLine("Please enter topics number:");
                int topicsNumber = Int32.Parse(Console.ReadLine());

                Console.WriteLine("Please enter vocabulary number:");
                int vocabularyNumber = Int32.Parse(Console.ReadLine());

                Console.WriteLine("Please enter the number of iteration:");
                int iteration = Int32.Parse(Console.ReadLine());

                if (modeltype == "d")
                    LDA(topicsNumber, vocabularyNumber, iteration);
                else if (modeltype == "a")
                    AT(topicsNumber, vocabularyNumber, iteration);
                else if (modeltype == "l")
                    LDATOT(topicsNumber, vocabularyNumber, iteration);
                else if (modeltype == "t")
                    ATOT(topicsNumber, vocabularyNumber, iteration);
                else if (modeltype == "b")
                    BT(topicsNumber, vocabularyNumber, iteration);

                Console.WriteLine("Please choose: Topic Modeling (M) or any key to exit:");
                choice = Console.ReadLine().ToLower();
            }
        }

        private static void BT(int topicsNumber, int vocabularyNumber, int iteration)
        {
            Parameters parameters = new Parameters(topicsNumber, vocabularyNumber, iteration);

            string[] docWordsFiles = GetLogFiles("*.docwords.txt", false, false);
            LinkedList<string> vocabList;
            string line;
            ExtractVocabulary(out vocabList, out line);
            string[] vocabArray = vocabList.ToArray();
            int V = vocabArray.Length;

            int M = docWordsFiles.Length;
            int[][] DW = new int[M][];

            for (int m = 0; m < M; m++)
            {
                LinkedList<int> docwords = ExtractInt(docWordsFiles, ref line, m, 0);
                DW[m] = docwords.ToArray();
            }

            BackgroundTopics bt = new BackgroundTopics(parameters.Beta, parameters.TopicsNumber, DW, vocabArray, parameters.Iterations);
            bt.BackGroundTopicsMCMC();

            for (int t = 0; t < parameters.TopicsNumber; t++)
            {
                Console.WriteLine("Please enter a name for the topic:");
                string topicname = Console.ReadLine().ToUpper();
                Console.WriteLine("Use the same size as Vocabulary Number: yes(y) or any button to use the previous entered number..");
                string vocabdecision = Console.ReadLine().ToLower();

                if (vocabdecision == "y")
                    vocabularyNumber = vocabArray.Length;


                StreamWriter resultFile = new StreamWriter("BT." + topicname + "." + vocabularyNumber + ".bgtopics.txt");

                resultFile.WriteLine("Topic:\t{0}\tSize:\t{1}", topicname, vocabularyNumber);
                resultFile.WriteLine();

                for (int v = 0; v < vocabularyNumber; v++)
                {
                    resultFile.WriteLine("{0}\t{1}", bt.PhiFT[t][v].Word, bt.PhiFT[t][v].Prob);
                }

                resultFile.Close();

                if (bt.SimilarityBTDW != null)
                {
                    resultFile = new StreamWriter("CosineSimilarity." + topicname + "." + ".similarity.txt");

                    for (int m = 0; m < M; m++)
                    {
                        FileInfo fi = new FileInfo(docWordsFiles[m]);
                        resultFile.WriteLine("{0}\tProbability:\t{1}", fi.Name, bt.SimilarityBTDW[m].Prob);
                    }

                    resultFile.Close();
                }

                if (bt.DocsSimilarity != null)
                {
                    resultFile = new StreamWriter("CosineSimilarityDocs.similarity.txt");

                    for (int i = 0; i < 4; i++)
                    {
                        for (int j = 0; j < 4; j++)
                        {
                            resultFile.WriteLine("{0}\tProbability:\t{1}",i+" docs with "+j, bt.DocsSimilarity[i][j].Prob);
                        }
                    }

                    resultFile.Close();
                }
            }
        }

        private static void LDA(int topicsNumber, int vocabularyNumber, int iteration)
        {
            Parameters parameters = new Parameters(topicsNumber, vocabularyNumber, iteration);
            StreamWriter resultFile = new StreamWriter("results." + parameters.TopicsNumber + ".LDAtopics.txt");
            resultFile.WriteLine("Started on {0} at {1}", DateTime.Now.ToShortDateString(), DateTime.Now.ToLongTimeString());

            string[] docWordsFiles = GetLogFiles("*.docwords.txt", false, false);
            string[] docFTopicsFiles = GetLogFiles("*.bgtopics.txt", true, false);

            LinkedList<string> vocabList;
            string line;
            ExtractVocabulary(out vocabList, out line);

            Result[][] PhiFT = ExtractBackTopics(docFTopicsFiles, ref vocabList);
            string[] vocabArray = vocabList.ToArray();
            int V = vocabArray.Length;

            int M = docWordsFiles.Length;
            int[][] DW = new int[M][];

            for (int m = 0; m < M; m++)
            {
                LinkedList<int> docwords = ExtractInt(docWordsFiles, ref line, m, 0);
                DW[m] = docwords.ToArray();
            }

            LDAGibbs lda = new LDAGibbs(parameters.Alpha, parameters.Beta, parameters.InvestigatorParameter, parameters.TopicsNumber, DW, PhiFT, vocabArray, parameters.Iterations);

            lda.MCMC();

            for (int t = 0; t < PhiFT.Length; t++)
            {
                StreamWriter kldTopics = new StreamWriter("results." + t + parameters.TopicsNumber + ".LDAtopics.txt");
                for (int d = 0; d < DW.Length; d++)
                {
                    kldTopics.WriteLine("Linear Interpolation for the selected Cybercrime Topics-Words {0} for the {1} document", t, docWordsFiles[d] + Environment.NewLine);
                    DistanceMetric metric = new DistanceMetric(M, PhiFT.Length, parameters.Beta, parameters.TopicsNumber, vocabArray.Length);
                    metric.TermFrequency(PhiFT, DW, d);

                    for (int w = 0; w < PhiFT[t].Length; w++)
                    {
                        kldTopics.WriteLine("{0}\t{1}\t{2}", w, PhiFT[t][w].Word, metric.TF[t][w]);
                    }
                    kldTopics.WriteLine();
                }

                kldTopics.WriteLine();
                kldTopics.WriteLine("Kullback–Leibler Divergence for this topic with all the documents");

                for (int d = 0; d < DW.Length; d++)
                {
                    kldTopics.WriteLine("Document {0}\t{1}", docWordsFiles[d], lda.KLDBTopicDocuments[d, t]);
                }

                kldTopics.WriteLine();
                kldTopics.WriteLine("Kullback–Leibler Divergence for this topic with all the topics");

                for (int k = 0; k < parameters.TopicsNumber; k++)
                {
                    kldTopics.WriteLine("Topic {0}\t{1}", k, lda.KLDTopics[t, k]);
                }

                kldTopics.WriteLine();
                kldTopics.WriteLine("Normalized Mutual Information for this topic with all the topics");
                for (int k = 0; k < parameters.TopicsNumber; k++)
                {
                    kldTopics.WriteLine("Topic {0}\t{1}", k, lda.NMITopics[t, k]);
                }
                kldTopics.Close();
            }

            for (int k = 0; k < parameters.TopicsNumber; k++)
            {
                resultFile.WriteLine("Topic {0}", k);
                for (int v = 0; v < parameters.MaxVocabNumber; v++)
                {
                    resultFile.WriteLine("{0}\t{1}\t{2}", v, lda.PhiTW[k][v].Word, lda.PhiTW[k][v].Prob);
                }
                resultFile.WriteLine();
            }

            for (int m = 0; m < M; m++)
            {
                resultFile.WriteLine(">> {0}", docWordsFiles[m]);

                for (int k = 0; k < parameters.TopicsNumber; k++)
                {
                    resultFile.WriteLine("\t{0}\t{1}", lda.ThetaDT[m][k].Word, lda.ThetaDT[m][k].Prob);
                }

                resultFile.WriteLine();
            }

            resultFile.WriteLine("Ended on {0} at {1}", DateTime.Now.ToShortDateString(), DateTime.Now.ToLongTimeString());
            resultFile.Close();

            Console.WriteLine("Calculate perplexity? yes(y) or any charecter to quit.");
            if (Console.ReadLine().ToLower() == "y")
            {
                double perplexity = PerplexityLDA(lda.PhiTW, parameters.TopicsNumber, vocabularyNumber, iteration);

                StreamWriter perplexityFile = new StreamWriter("results.Perplexity.LDA.txt");
                perplexityFile.WriteLine("Perplexity for topics:{0} is {1}", topicsNumber, perplexity);
                perplexityFile.Close();
            }
        }

        private static double PerplexityLDA(Result[][] PhiTW, int TopicsNumber, int vocabularyNumber, int iteration)
        {
            Parameters parameters = new Parameters(TopicsNumber, vocabularyNumber, iteration);

            string[] docWordsTestFiles = GetLogFiles("*.docwords.txt", false, true);

            LinkedList<string> vocabList;
            string line = String.Empty;
            ExtractVocabulary(out vocabList, out line);

            string[] vocabArray = vocabList.ToArray();
            int V = vocabArray.Length;

            int M = docWordsTestFiles.Length;
            int[][] DW = new int[M][];

            for (int m = 0; m < M; m++)
            {
                LinkedList<int> docwords = ExtractInt(docWordsTestFiles, ref line, m, 0);
                DW[m] = docwords.ToArray();
            }

            LDAGibbs lda = new LDAGibbs(parameters.Alpha, parameters.Beta, parameters.TopicsNumber, DW, vocabArray, parameters.Iterations);

            DateTime systemDate = DateTime.Now;
            Console.WriteLine("Started at:" +systemDate);

            lda.MCMCPerplexity();

            ArrayList uniquewords = new ArrayList();
            int toTalwords = 0;

            for (int d = 0; d < DW.Length; d++)
                for (int n = 0; n < DW[d].Length; n++)
                {
                    if (!uniquewords.Contains(DW[d][n]))
                        uniquewords.Add(DW[d][n]);
                    toTalwords++;
                }

            double log = 0;

            for (int d = 0; d < M; d++)
            {
                for (int v = 0; v < uniquewords.Count; v++)
                {
                    int numberofUniqueTerms = 0;
                    double probability = 0;

                    for (int n = 0; n < DW[d].Length; n++)
                    {
                        if (DW[d][n] == Convert.ToInt32(uniquewords[v]))
                            numberofUniqueTerms++;
                    }

                    for (int k = 0; k < TopicsNumber; k++)
                    {
                        probability += lda.ThetaDT[d][k].Prob * PhiTW[k][Convert.ToInt32(uniquewords[v])].Prob;
                    }
                    log += Math.Log(probability) * numberofUniqueTerms;
                }
            }

            double perplexity = 0;
            perplexity = Math.Exp(-(log / toTalwords));
            Console.WriteLine(perplexity);

            systemDate = DateTime.Now;
            Console.WriteLine("Finished at: "+systemDate);

            return perplexity;
        }

        private static void LDATOT(int topicsNumber, int vocabularyNumber, int iteration)
        {
            Parameters parameters = new Parameters(topicsNumber, vocabularyNumber, iteration);
            StreamWriter resultFile = new StreamWriter("results." + parameters.TopicsNumber + ".LDATOTtopics.txt");
            resultFile.WriteLine("Started on {0} at {1}", DateTime.Now.ToShortDateString(), DateTime.Now.ToLongTimeString());

            string[] docWordsFiles = GetLogFiles("*.docwords.txt", false, false);
            string[] docTimesFiles = GetLogFiles("*.times.txt", false, false);
            string[] docWordsTimesFiles = GetLogFiles("*.wordstimes.txt", false, false);
            string[] docFTopicsFiles = GetLogFiles("*.bgtopics.txt", true, false);

            LinkedList<string> vocabList;
            string line;
            ExtractVocabulary(out vocabList, out line);

            Result[][] PhiFT = ExtractBackTopics(docFTopicsFiles, ref vocabList);
            string[] vocabArray = vocabList.ToArray();
            int V = vocabArray.Length;

            int M = docWordsFiles.Length;
            int[][] DW = new int[M][];
            double[][] WT = new double[M][];

            for (int m = 0; m < M; m++)
            {
                LinkedList<int> docwords = ExtractInt(docWordsFiles, ref line, m, 0);
                DW[m] = docwords.ToArray();
                LinkedList<double> docWordsTimes = ExtractDouble(docWordsTimesFiles, ref line, m, 1);
                WT[m] = docWordsTimes.ToArray();
            }

            LinkedList<double> doctimes = ExtractDouble(docTimesFiles, ref line, 1);
            double[] DT = doctimes.ToArray();

            LDATOTGibbs ldatot = new LDATOTGibbs(parameters.Alpha, parameters.Beta, parameters.InvestigatorParameter, parameters.TopicsNumber, DW, DT, WT, PhiFT, vocabArray, parameters.Iterations);

            ldatot.MCMC();

            for (int t = 0; t < PhiFT.Length; t++)
            {
                StreamWriter kldTopics = new StreamWriter("results." + t + parameters.TopicsNumber + ".LDATOTtopics.txt");
                for (int d = 0; d < DW.Length; d++)
                {
                    kldTopics.WriteLine("Linear Interpolation for the selected Cybercrime Topics-Words {0} for the {1} document", t, docWordsFiles[d] + Environment.NewLine);
                    DistanceMetric metric = new DistanceMetric(M, PhiFT.Length, parameters.Beta, parameters.TopicsNumber, vocabArray.Length);
                    metric.TermFrequency(PhiFT, DW, d);

                    for (int w = 0; w < PhiFT[t].Length; w++)
                    {
                        kldTopics.WriteLine("{0}\t{1}\t{2}", w, PhiFT[t][w].Word, metric.TF[t][w]);
                    }
                    kldTopics.WriteLine();


                    kldTopics.WriteLine("Active time for this document :");
                    for (int s = 0; s < DT.Length; s++)
                    {
                        kldTopics.WriteLine("{0}\t{1}", Math.Round(DT[s] * 23.59, 2), ldatot.DOT[d][s].Prob);
                    }
                    kldTopics.WriteLine();
                }

                for (int s = 0; s < DT.Length; s++)
                {
                    kldTopics.WriteLine("{0}\t{1}", Math.Round(DT[s] * 23.59, 2), ldatot.BetaFTOT[t][s].Prob);
                }

                kldTopics.WriteLine();
                kldTopics.WriteLine("Kullback–Leibler Divergence for this topic with all the documents");

                for (int d = 0; d < DW.Length; d++)
                {
                    kldTopics.WriteLine("Document {0}\t{1}", docWordsFiles[d], ldatot.KLDBTopicDocuments[d, t]);
                }

                kldTopics.WriteLine();
                kldTopics.WriteLine("Kullback–Leibler Divergence for this topic with all the topics");

                for (int k = 0; k < parameters.TopicsNumber; k++)
                {
                    kldTopics.WriteLine("Topic {0}\t{1}", k, ldatot.KLDTopics[t, k]);
                }

                kldTopics.WriteLine();
                kldTopics.WriteLine("Normalized Mutual Information for this topic with all the topics");
                for (int k = 0; k < parameters.TopicsNumber; k++)
                {
                    kldTopics.WriteLine("Topic {0}\t{1}", k, ldatot.NMITopics[t, k]);
                }
                kldTopics.Close();
            }

            for (int k = 0; k < parameters.TopicsNumber; k++)
            {
                resultFile.WriteLine("Topic {0}", k);
                for (int v = 0; v < parameters.MaxVocabNumber; v++)
                {
                    resultFile.WriteLine("{0}\t{1}\t{2}", v, ldatot.PhiTW[k][v].Word, ldatot.PhiTW[k][v].Prob);
                }
                for (int t = 0; t < DT.Length; t++)
                {
                    resultFile.WriteLine("{0}\t{1}", Math.Round(DT[t] * 23.59, 2), ldatot.BetaTOT[k][t].Prob);
                }
                resultFile.WriteLine();
            }

            for (int m = 0; m < M; m++)
            {
                resultFile.WriteLine(">> {0}", docWordsFiles[m]);

                for (int k = 0; k < parameters.TopicsNumber; k++)
                {
                    resultFile.WriteLine("\t{0}\t{1}", ldatot.ThetaDT[m][k].Word, ldatot.ThetaDT[m][k].Prob);
                }

                resultFile.WriteLine();
            }

            resultFile.WriteLine("Ended on {0} at {1}", DateTime.Now.ToShortDateString(), DateTime.Now.ToLongTimeString());
            resultFile.Close();

            Console.WriteLine("Calculate perplexity? yes(y) or any charecter to quit.");
            if (Console.ReadLine().ToLower() == "y")
            {
                double perplexity = PerplexityLDA(ldatot.PhiTW, parameters.TopicsNumber, vocabularyNumber, iteration);

                StreamWriter perplexityFile = new StreamWriter("results.Perplexity.LDATOT.txt");
                perplexityFile.WriteLine("Perplexity for topics:{0} is {1}", topicsNumber, perplexity);
                perplexityFile.Close();
            }
        }

        private static double PerplexityLDATOT(Result[][] PhiTW, int TopicsNumber, int vocabularyNumber, int iteration)
        {
            Parameters parameters = new Parameters(TopicsNumber, vocabularyNumber, iteration);

            string[] docWordsTestFiles = GetLogFiles("*.docwords.txt", false, true);
            string[] docTimesFiles = GetLogFiles("*.times.txt", false, true);
            string[] docWordsTimesFiles = GetLogFiles("*.wordstimes.txt", false, true);

            LinkedList<string> vocabList;
            string line = String.Empty;
            ExtractVocabulary(out vocabList, out line);

            string[] vocabArray = vocabList.ToArray();
            int V = vocabArray.Length;

            int M = docWordsTestFiles.Length;
            int[][] DW = new int[M][];
            double[][] WT = new double[M][];

            for (int m = 0; m < M; m++)
            {
                LinkedList<int> docwords = ExtractInt(docWordsTestFiles, ref line, m, 0);
                DW[m] = docwords.ToArray();
                LinkedList<double> docWordsTimes = ExtractDouble(docWordsTimesFiles, ref line, m, 1);
                WT[m] = docWordsTimes.ToArray();
            }

            LinkedList<double> doctimes = ExtractDouble(docTimesFiles, ref line, 1);
            double[] DT = doctimes.ToArray();


            LDATOTGibbs ldatot = new LDATOTGibbs(parameters.Alpha, parameters.Beta, parameters.TopicsNumber, DW, DT, WT, vocabArray, parameters.Iterations);
            DateTime systemDate = DateTime.Now;
            Console.WriteLine("Started at:" + systemDate);
            ldatot.MCMCPerplexity();

            ArrayList uniquewords = new ArrayList();
            int toTalwords = 0;

            for (int d = 0; d < DW.Length; d++)
                for (int n = 0; n < DW[d].Length; n++)
                {
                    if (!uniquewords.Contains(DW[d][n]))
                        uniquewords.Add(DW[d][n]);
                    toTalwords++;

                }

            double log = 0;

            for (int d = 0; d < M; d++)
            {
                for (int v = 0; v < uniquewords.Count; v++)
                {
                    int numberofUniqueTerms = 0;
                    double probability = 0;

                    for (int n = 0; n < DW[d].Length; n++)
                    {
                        if (DW[d][n] == Convert.ToInt32(uniquewords[v]))
                            numberofUniqueTerms++;
                    }

                    for (int k = 0; k < TopicsNumber; k++)
                    {
                        probability += ldatot.ThetaDT[d][k].Prob * PhiTW[k][Convert.ToInt32(uniquewords[v])].Prob;
                    }

                    log += Math.Log(probability) * numberofUniqueTerms;
                }
            }

            double perplexity = 0;
            perplexity = Math.Exp(-(log / toTalwords));
            Console.WriteLine(perplexity);
            systemDate = DateTime.Now;
            Console.WriteLine("Finished at: " + systemDate);
            return perplexity;
        }

        private static void AT(int topicsNumber, int vocabularyNumber, int iteration)
        {

            Parameters parameters = new Parameters(topicsNumber, vocabularyNumber, iteration);
            StreamWriter resultFile = new StreamWriter("results." + parameters.TopicsNumber + ".ATtopics.txt");
            resultFile.WriteLine("Started on {0} at {1}", DateTime.Now.ToShortDateString(), DateTime.Now.ToShortTimeString());

            string[] docWordsFiles = GetLogFiles("*.docwords.txt", false, false);
            string[] docAuthorsFiles = GetLogFiles("*.authors.txt", false, false);
            string[] docWordsAuthorsFiles = GetLogFiles("*.wordsauthors.txt", false, false);
            string[] docFTopicsFiles = GetLogFiles("*.bgtopics.txt", true, false);

            LinkedList<string> vocabList;
            string line;
            ExtractVocabulary(out vocabList, out line);

            Result[][] PhiFT = ExtractBackTopics(docFTopicsFiles, ref vocabList);
            string[] vocabArray = vocabList.ToArray();
            int V = vocabArray.Length;

            int M = docWordsFiles.Length;
            int[][] DW = new int[M][];
            string[][] DA = new string[M][];
            int[][] WA = new int[M][];


            for (int m = 0; m < M; m++)
            {
                LinkedList<int> docwords = ExtractInt(docWordsFiles, ref line, m, 0);
                DW[m] = docwords.ToArray();
                LinkedList<string> docauthors = ExtractString(docAuthorsFiles, ref line, m, 1);
                DA[m] = docauthors.ToArray();
                docwords = ExtractInt(docWordsAuthorsFiles, ref line, m, 1);
                WA[m] = docwords.ToArray();

            }

            ATGibbs authorTopic = new ATGibbs(parameters.Alpha, parameters.Beta, parameters.InvestigatorParameter, parameters.TopicsNumber, DW, DA, WA, PhiFT, vocabArray, parameters.Iterations);

            authorTopic.MCMC();

            for (int t = 0; t < PhiFT.Length; t++)
            {
                StreamWriter kldTopics = new StreamWriter("results." + t + parameters.TopicsNumber + ".ATtopics.txt");
                for (int d = 0; d < DW.Length; d++)
                {
                    kldTopics.WriteLine("Linear Interpolation for the selected Cybercrime Topics-Words {0} for the {1} document", t, docWordsFiles[d] + Environment.NewLine);
                    DistanceMetric metric = new DistanceMetric(M, PhiFT.Length, parameters.Beta, parameters.TopicsNumber, vocabArray.Length);
                    metric.TermFrequency(PhiFT, DW, d);

                    for (int w = 0; w < PhiFT[t].Length; w++)
                    {
                        kldTopics.WriteLine("{0}\t{1}\t{2}", w, PhiFT[t][w].Word, metric.TF[t][w]);
                    }
                    kldTopics.WriteLine();
                }

                kldTopics.WriteLine();
                kldTopics.WriteLine("Kullback–Leibler Divergence for this topic with all the documents");

                for (int d = 0; d < DW.Length; d++)
                {
                    kldTopics.WriteLine("Document {0}\t{1}", docWordsFiles[d], authorTopic.KLDBTopicDocuments[d, t]);
                }

                kldTopics.WriteLine();
                kldTopics.WriteLine("Kullback–Leibler Divergence for this topic with all the topics");

                for (int k = 0; k < parameters.TopicsNumber; k++)
                {
                    kldTopics.WriteLine("Topic {0}\t{1}", k, authorTopic.KLDTopics[t, k]);
                }

                kldTopics.WriteLine();
                kldTopics.WriteLine("Normalized Mutual Information for this topic with all the topics");
                for (int k = 0; k < parameters.TopicsNumber; k++)
                {
                    kldTopics.WriteLine("Topic {0}\t{1}", k, authorTopic.NMITopics[t, k]);
                }

                kldTopics.Close();
            }

            for (int k = 0; k < parameters.TopicsNumber; k++)
            {
                resultFile.WriteLine("Topic {0}", k);
                for (int v = 0; v < parameters.MaxVocabNumber; v++)
                {
                    resultFile.WriteLine("{0}\t{1}\t{2}", v, authorTopic.PhiTW[k][v].Word, authorTopic.PhiTW[k][v].Prob);
                }
                resultFile.WriteLine(GetHighestAuthorPerTopic(authorTopic.ThetaAT, DA, M, k, 2));

                resultFile.WriteLine();
            }

            for (int m = 0; m < M; m++)
            {
                resultFile.WriteLine(">> {0}", docWordsFiles[m]);

                for (int k = 0; k < parameters.TopicsNumber; k++)
                {
                    resultFile.WriteLine("\t{0}\t{1}", authorTopic.ThetaDT[m][k].Word, authorTopic.ThetaDT[m][k].Prob);
                }

                resultFile.WriteLine();

                for (int l = 0; l < DA[m].Length; l++)
                {
                    resultFile.WriteLine("\tAuthor: {0}", DA[m][l]);
                    for (int k = 0; k < parameters.TopicsNumber; k++)
                    {
                        resultFile.WriteLine("\t\tTopic: {0}\t Probablity: {1}", authorTopic.ThetaAT[m][l, k].Topic, authorTopic.ThetaAT[m][l, k].Prob.ToString());
                    }
                    resultFile.WriteLine();
                }
                resultFile.WriteLine();
            }
            resultFile.WriteLine("Ended on {0} at {1}", DateTime.Now.ToShortDateString(), DateTime.Now.ToLongTimeString());
            resultFile.Close();

            Console.WriteLine("Calculate perplexity? yes(y) or any charecter to quit.");
            if (Console.ReadLine().ToLower() == "y")
            {
                double perplexity = PerplexityLDA(authorTopic.PhiTW, parameters.TopicsNumber, vocabularyNumber, iteration);

                StreamWriter perplexityFile = new StreamWriter("results.Perplexity.AT.txt");
                perplexityFile.WriteLine("Perplexity for topics:{0} is {1}", topicsNumber, perplexity);
                perplexityFile.Close();
            }
        }

        private static double PerplexityAT(Result[][] PhiTW, int TopicsNumber, int vocabularyNumber, int iteration)
        {
            Parameters parameters = new Parameters(TopicsNumber, vocabularyNumber, iteration);

            string line = String.Empty;

            string[] docWordsTestFiles = GetLogFiles("*.docwords.txt", false, true);
            string[] docAuthorsTestFiles = GetLogFiles("*.authors.txt", false, true);
            string[] docWordsAuthorsTestFiles = GetLogFiles("*.wordsauthors.txt", false, true);

            int totalauthors = 0;
            int M = docWordsTestFiles.Length;
            int[][] DW = new int[M][];
            string[][] DA = new string[M][];
            int[][] WA = new int[M][];

            LinkedList<string> vocabList;
            ExtractVocabulary(out vocabList, out line);
            string[] vocabArray = vocabList.ToArray();
            int V = vocabArray.Length;

            for (int m = 0; m < M; m++)
            {
                LinkedList<int> docwords = ExtractInt(docWordsTestFiles, ref line, m, 0);
                DW[m] = docwords.ToArray();
                LinkedList<string> docauthors = ExtractString(docAuthorsTestFiles, ref line, m, 1);
                DA[m] = docauthors.ToArray();
                totalauthors += DA[m].Length;
                docwords = ExtractInt(docWordsAuthorsTestFiles, ref line, m, 1);
                WA[m] = docwords.ToArray();
            }

            ATGibbs authorTopic = new ATGibbs(parameters.Alpha, parameters.Beta, parameters.TopicsNumber, DW, DA, WA, vocabArray, parameters.Iterations);
            DateTime systemDate = DateTime.Now;
            Console.WriteLine("Started at:" + systemDate);
            authorTopic.MCMCPerplexity();

            ArrayList uniquewords = new ArrayList();
            int toTalwords = 0;

            for (int d = 0; d < DW.Length; d++)
                for (int n = 0; n < DW[d].Length; n++)
                {
                    if (!uniquewords.Contains(DW[d][n]))
                        uniquewords.Add(DW[d][n]);
                    toTalwords++;
                }

            double log = 0;

            for (int d = 0; d < M; d++)
            {
                for (int v = 0; v < uniquewords.Count; v++)
                {
                    int numberofUniqueTerms = 0;
                    double probability = 0;

                    for (int n = 0; n < DW[d].Length; n++)
                    {
                        if (DW[d][n] == Convert.ToInt32(uniquewords[v]))
                            numberofUniqueTerms++;
                    }

                    for (int x = 0; x < DA[d].Length; x++)
                    {

                        for (int k = 0; k < TopicsNumber; k++)
                        {
                            probability += authorTopic.ThetaAT[d][x, k].Prob * PhiTW[k][Convert.ToInt32(uniquewords[v])].Prob;
                        }
                    }
                    //log += (Math.Log(probability) * numberofUniqueTerms) / DA[d].Length;
                    log += (Math.Log(probability) * numberofUniqueTerms);
                }
            }

            double perplexity = Math.Exp(-(log / toTalwords));
            Console.WriteLine(perplexity);
            systemDate = DateTime.Now;
            Console.WriteLine("Finished at: " + systemDate);
            return perplexity;
        }

        private static void ATOT(int topicsNumber, int vocabularyNumber, int iteration)
        {
            Parameters parameters = new Parameters(topicsNumber, vocabularyNumber, iteration);
            StreamWriter resultFile = new StreamWriter("results." + parameters.TopicsNumber + ".ATOTtopics.txt");
            resultFile.WriteLine("Started on {0} at {1}", DateTime.Now.ToShortDateString(), DateTime.Now.ToShortTimeString());

            string[] docWordsFiles = GetLogFiles("*.docwords.txt", false, false);
            string[] docAuthorsFiles = GetLogFiles("*.authors.txt", false, false);
            string[] docWordsAuthorsFiles = GetLogFiles("*.wordsauthors.txt", false, false);
            string[] docTimesFiles = GetLogFiles("*.times.txt", false, false);
            string[] docWordsTimesFiles = GetLogFiles("*.wordstimes.txt", false, false);
            string[] docFTopicsFiles = GetLogFiles("*.bgtopics.txt", true, false);

            LinkedList<string> vocabList;
            string line;
            ExtractVocabulary(out vocabList, out line);

            Result[][] PhiFT = ExtractBackTopics(docFTopicsFiles, ref vocabList);
            string[] vocabArray = vocabList.ToArray();
            int V = vocabArray.Length;

            int M = docWordsFiles.Length;
            int[][] DW = new int[M][];
            string[][] DA = new string[M][];
            int[][] WA = new int[M][];
            double[][] WT = new double[M][];

            for (int m = 0; m < M; m++)
            {
                LinkedList<int> docwords = ExtractInt(docWordsFiles, ref line, m, 0);
                DW[m] = docwords.ToArray();
                LinkedList<string> docauthors = ExtractString(docAuthorsFiles, ref line, m, 1);
                DA[m] = docauthors.ToArray();
                docwords = ExtractInt(docWordsAuthorsFiles, ref line, m, 1);
                WA[m] = docwords.ToArray();
                LinkedList<double> docWordsTimes = ExtractDouble(docWordsTimesFiles, ref line, m, 1);
                WT[m] = docWordsTimes.ToArray();
            }

            LinkedList<double> doctimes = ExtractDouble(docTimesFiles, ref line, 1);
            double[] DT = doctimes.ToArray();

            ATOTGibbs authorTopic = new ATOTGibbs(parameters.Alpha, parameters.Beta, parameters.InvestigatorParameter, parameters.TopicsNumber, DW, DA, WA, DT, WT, PhiFT, vocabArray, parameters.Iterations);

            authorTopic.MCMC();         

            for (int t = 0; t < PhiFT.Length; t++)
            {
                StreamWriter kldTopics = new StreamWriter("results." + t + parameters.TopicsNumber + ".ATOTtopics.txt");
                for (int d = 0; d < DW.Length; d++)
                {
                    kldTopics.WriteLine("Linear Interpolation for the selected Cybercrime Topics-Words {0} for the {1} document", t, docWordsFiles[d] + Environment.NewLine);
                    DistanceMetric metric = new DistanceMetric(M, PhiFT.Length, parameters.Beta, parameters.TopicsNumber, vocabArray.Length);
                    metric.TermFrequency(PhiFT, DW, d);

                    for (int w = 0; w < PhiFT[t].Length; w++)
                    {
                        kldTopics.WriteLine("{0}\t{1}\t{2}", w, PhiFT[t][w].Word, metric.TF[t][w]);
                    }
                    kldTopics.WriteLine();

                    kldTopics.WriteLine("Active time for this document :");
                    for (int s = 0; s < DT.Length; s++)
                    {
                        kldTopics.WriteLine("{0}\t{1}", Math.Round(DT[s] * 23.59, 2), authorTopic.DOT[d][s].Prob);
                    }
                    kldTopics.WriteLine();
                }

                kldTopics.WriteLine();
                for (int s = 0; s < DT.Length; s++)
                {
                    kldTopics.WriteLine("{0}\t{1}", Math.Round(DT[s] * 23.59, 2), authorTopic.BetaFTOT[t][s].Prob);
                }

                kldTopics.WriteLine();
                kldTopics.WriteLine("Kullback–Leibler Divergence for this topic with all the documents");

                for (int d = 0; d < DW.Length; d++)
                {
                    kldTopics.WriteLine("Document {0}\t{1}", docWordsFiles[d], authorTopic.KLDBTopicDocuments[d, t]);
                }

                kldTopics.WriteLine();
                kldTopics.WriteLine("Kullback–Leibler Divergence for this topic with all the topics");

                for (int k = 0; k < parameters.TopicsNumber; k++)
                {
                    kldTopics.WriteLine("Topic {0}\t{1}", k, authorTopic.KLDTopics[t, k]);
                }

                kldTopics.WriteLine();
                kldTopics.WriteLine("Normalized Mutual Information for this topic with all the topics");
                for (int k = 0; k < parameters.TopicsNumber; k++)
                {
                    kldTopics.WriteLine("Topic {0}\t{1}", k, authorTopic.NMITopics[t, k]);
                }

                kldTopics.Close();
            }

            for (int k = 0; k < parameters.TopicsNumber; k++)
            {
                resultFile.WriteLine("Topic {0}", k);
                for (int v = 0; v < parameters.MaxVocabNumber; v++)
                {
                    resultFile.WriteLine("{0}\t{1}\t{2}", v, authorTopic.PhiTW[k][v].Word, authorTopic.PhiTW[k][v].Prob);
                }
                resultFile.WriteLine(GetHighestAuthorPerTopic(authorTopic.ThetaAT, DA, M, k, 2));
                for (int t = 0; t < DT.Length; t++)
                {
                    resultFile.WriteLine("{0}\t{1}", Math.Round(DT[t] * 23.59, 2), authorTopic.BetaTOT[k][t].Prob);
                }
                resultFile.WriteLine();
            }

            for (int m = 0; m < M; m++)
            {
                resultFile.WriteLine(">> {0}", docWordsFiles[m]);

                for (int k = 0; k < parameters.TopicsNumber; k++)
                {
                    resultFile.WriteLine("\t{0}\t{1}", authorTopic.ThetaDT[m][k].Word, authorTopic.ThetaDT[m][k].Prob);
                }

                resultFile.WriteLine();

                for (int l = 0; l < DA[m].Length; l++)
                {
                    resultFile.WriteLine("\tAuthor: {0}", DA[m][l]);
                    for (int k = 0; k < parameters.TopicsNumber; k++)
                    {
                        resultFile.WriteLine("\t\tTopic: {0}\t Probablity: {1}", authorTopic.ThetaAT[m][l, k].Topic, authorTopic.ThetaAT[m][l, k].Prob.ToString());
                    }
                    resultFile.WriteLine();

                    for (int s = 0; s < DT.Length; s++)
                    {
                        resultFile.WriteLine("{0}\t{1}", Math.Round(DT[s] * 23.59, 2), authorTopic.ATOT[m][l, s].Prob);
                    }
                    resultFile.WriteLine();
                }
                resultFile.WriteLine();
            }
            resultFile.WriteLine("Ended on {0} at {1}", DateTime.Now.ToShortDateString(), DateTime.Now.ToLongTimeString());
            resultFile.Close();

            Console.WriteLine("Calculate perplexity? yes(y) or any charecter to quit.");
            if (Console.ReadLine().ToLower() == "y")
            {
                double perplexity = PerplexityATOT(authorTopic.PhiTW, parameters.TopicsNumber, vocabularyNumber, iteration);

                StreamWriter perplexityFile = new StreamWriter("results.Perplexity.ATOT.txt");
                perplexityFile.WriteLine("Perplexity for topics:{0} is {1}", topicsNumber, perplexity);
                perplexityFile.Close();
            }
        }

        private static double PerplexityATOT(Result[][] PhiTW, int TopicsNumber, int vocabularyNumber, int iteration)
        {
            Parameters parameters = new Parameters(TopicsNumber, vocabularyNumber, iteration);

            string line = String.Empty;

            string[] docWordsTestFiles = GetLogFiles("*.docwords.txt", false, true);
            string[] docAuthorsTestFiles = GetLogFiles("*.authors.txt", false, true);
            string[] docWordsAuthorsTestFiles = GetLogFiles("*.wordsauthors.txt", false, true);
            string[] docTimesTestFiles = GetLogFiles("*.times.txt", false, true);
            string[] docWordsTimesTestFiles = GetLogFiles("*.wordstimes.txt", false, true);

            int totalauthors = 0;
            int M = docWordsTestFiles.Length;
            int[][] DW = new int[M][];
            string[][] DA = new string[M][];
            int[][] WA = new int[M][];
            double[][] WT = new double[M][];

            LinkedList<string> vocabList;
            ExtractVocabulary(out vocabList, out line);
            string[] vocabArray = vocabList.ToArray();
            int V = vocabArray.Length;

            for (int m = 0; m < M; m++)
            {
                LinkedList<int> docwords = ExtractInt(docWordsTestFiles, ref line, m, 0);
                DW[m] = docwords.ToArray();
                LinkedList<string> docauthors = ExtractString(docAuthorsTestFiles, ref line, m, 1);
                DA[m] = docauthors.ToArray();
                totalauthors += DA[m].Length;
                docwords = ExtractInt(docWordsAuthorsTestFiles, ref line, m, 1);
                WA[m] = docwords.ToArray();
                LinkedList<double> docWordsTimes = ExtractDouble(docWordsTimesTestFiles, ref line, m, 1);
                WT[m] = docWordsTimes.ToArray();
            }

            LinkedList<double> doctimes = ExtractDouble(docTimesTestFiles, ref line, 1);
            double[] DT = doctimes.ToArray();


            ATOTGibbs authorTopic = new ATOTGibbs(parameters.Alpha, parameters.Beta, parameters.TopicsNumber, DW, DA, WA, DT, WT, vocabArray, parameters.Iterations);

            DateTime systemDate = DateTime.Now;
            Console.WriteLine("Started at:" + systemDate);

            authorTopic.MCMCPerplexity();

            ArrayList uniquewords = new ArrayList();
            int toTalwords = 0;

            for (int d = 0; d < DW.Length; d++)
                for (int n = 0; n < DW[d].Length; n++)
                {
                    if (!uniquewords.Contains(DW[d][n]))
                        uniquewords.Add(DW[d][n]);
                    toTalwords++;
                }

            double log = 0;

            for (int d = 0; d < M; d++)
            {
                for (int v = 0; v < uniquewords.Count; v++)
                {
                    int numberofUniqueTerms = 0;
                    double probability = 0;

                    for (int n = 0; n < DW[d].Length; n++)
                    {
                        if (DW[d][n] == Convert.ToInt32(uniquewords[v]))
                            numberofUniqueTerms++;
                    }

                    for (int x = 0; x < DA[d].Length; x++)
                    {

                        for (int k = 0; k < TopicsNumber; k++)
                        {
                            probability += authorTopic.ThetaAT[d][x, k].Prob * PhiTW[k][Convert.ToInt32(uniquewords[v])].Prob;
                        }
                    }
                    //log += (Math.Log(probability) * numberofUniqueTerms) / DA[d].Length;
                    log += (Math.Log(probability) * numberofUniqueTerms);
                }
            }

            double perplexity = Math.Exp(-(log / toTalwords));
            Console.WriteLine(perplexity);
            systemDate = DateTime.Now;
            Console.WriteLine("Finished at: " + systemDate);
            return perplexity;
        }

        private static int ArrayIndexofAuthor(Result[][,] ThetaAT, string author, int K, ref int D)
        {
            for (int d = 0; d < ThetaAT.Length; d++)
            {
                for (int i = 0; i < (ThetaAT[d].Length) / K; i++)
                {
                    if (author == ThetaAT[d][i, 0].Author)
                    {
                        D = d;
                        return i;
                    }
                }
            }
            return -1;
        }

        private static LinkedList<int> ExtractInt(string[] docFiles, ref string line, int m, int index)
        {
            LinkedList<int> docwords = new LinkedList<int>();
            StreamReader sr = new StreamReader(docFiles[m]);
            line = sr.ReadLine();
            while (line != null)
            {
                if (line.Equals(""))
                {
                    line = sr.ReadLine();
                    continue;
                }
                string[] tokens = line.Split('\t');
                docwords.AddLast(int.Parse(tokens[index]));
                line = sr.ReadLine();
            }
            sr.Close();
            return docwords;
        }

        private static LinkedList<double> ExtractDouble(string[] docFiles, ref string line, int m, int index)
        {
            LinkedList<double> docwords = new LinkedList<double>();
            StreamReader sr = new StreamReader(docFiles[m]);
            line = sr.ReadLine();
            while (line != null)
            {
                if (line.Equals(""))
                {
                    line = sr.ReadLine();
                    continue;
                }
                string[] tokens = line.Split('\t');
                docwords.AddLast(double.Parse(tokens[index]));
                line = sr.ReadLine();
            }
            sr.Close();
            return docwords;
        }

        private static LinkedList<double> ExtractDouble(string[] docFiles, ref string line, int index)
        {
            ArrayList temparrayTimes = new ArrayList();

            for (int m = 0; m < docFiles.Length; m++)
            {
                StreamReader sr = new StreamReader(docFiles[m]);
                line = sr.ReadLine();
                while (line != null)
                {
                    if (line.Equals(""))
                    {
                        line = sr.ReadLine();
                        continue;
                    }
                    string[] tokens = line.Split('\t');
                    double tokensTime = double.Parse(tokens[index]);
                    if (!temparrayTimes.Contains(tokensTime))
                    {
                        temparrayTimes.Add(tokensTime);
                    }
                    line = sr.ReadLine();
                }
                sr.Close();
            }
            temparrayTimes.Sort();

            LinkedList<double> docwords = new LinkedList<double>();

            for (int i = 0; i < temparrayTimes.Count; i++)
                docwords.AddLast(double.Parse(temparrayTimes[i].ToString()));

            return docwords;
        }

        private static LinkedList<string> ExtractString(string[] docFiles, ref string line, int m, int index)
        {
            LinkedList<string> docwords = new LinkedList<string>();
            StreamReader sr = new StreamReader(docFiles[m]);
            line = sr.ReadLine();
            while (line != null)
            {
                if (line.Equals(""))
                {
                    line = sr.ReadLine();
                    continue;
                }
                string[] tokens = line.Split('\t');
                docwords.AddLast(tokens[index]);
                line = sr.ReadLine();
            }
            sr.Close();
            return docwords;
        }

        private static Result[][] ExtractBackTopics(string[] docFiles, ref LinkedList<string> vocabList)
        {
            Result[][] PhiFT = new Result[docFiles.Length][];

            for (int i = 0; i < docFiles.Length; i++)
            {
                StreamReader sr = new StreamReader(docFiles[i]);
                string line = sr.ReadLine();
                int index = 0;

                while (line != null)
                {
                    if (line.Equals(""))
                    {
                        line = sr.ReadLine();
                        continue;
                    }
                    string[] tokens = line.Split(' ', '\t');

                    if (tokens[0] == "Topic:")
                    {
                        line = sr.ReadLine();
                        PhiFT[i] = new Result[Int32.Parse(tokens[3])];
                        continue;
                    }
                    else
                    {
                        if (!vocabList.Contains(tokens[0].ToLower()))
                        {
                            vocabList.AddLast(tokens[0].ToLower());
                            PhiFT[i][index] = new Result(tokens[0].ToLower(), double.Parse(tokens[1]), Array.IndexOf(vocabList.ToArray(), tokens[0].ToLower()));
                            line = sr.ReadLine();
                            index++;
                        }
                        else
                        {
                            PhiFT[i][index] = new Result(tokens[0].ToLower(), double.Parse(tokens[1]), Array.IndexOf(vocabList.ToArray(), tokens[0].ToLower()));
                            line = sr.ReadLine();
                            index++;
                        }
                    }
                }

                sr.Close();
            }

            return PhiFT;
        }

        private static void ExtractVocabulary(out LinkedList<string> vocabList, out string line)
        {
            vocabList = new LinkedList<string>();
            StreamReader dictFile = new StreamReader(new TextPath().Vocabpath);

            line = dictFile.ReadLine();
            while (line != null)
            {
                if (line.Equals(""))
                {
                    line = dictFile.ReadLine();
                    continue;
                }

                vocabList.AddLast(line.Trim());
                line = dictFile.ReadLine();
            }

            dictFile.Close();
        }

        private static string[] GetLogFiles(string path, bool bgTopics, bool testfiles)
        {
            Contract.Ensures(Contract.Result<string[]>() != null && Contract.Result<string[]>().Length != 0, "result is null or empty.");

            FileInfo[] fi;

            if (bgTopics && !testfiles)
                fi = new DirectoryInfo(new TextPath().Bgtopicspath).GetFiles(path);
            else if (!bgTopics && testfiles)
                fi = new DirectoryInfo(new TextPath().Testfilespath).GetFiles(path);
            else
                fi = new DirectoryInfo(new TextPath().Indexpath).GetFiles(path);

            string[] files = new string[fi.Length];

            for (int i = 0; i < fi.Length; i++)
            { files[i] = fi[i].FullName; }

            return files;
        }

        private static string GetHighestAuthorPerTopic(Result[][,] ThetaAT, string[][] DA, int DocumentsNumber, int TopicIndex, int MaxAuthor)
        {
            int sumauthors = 0;

            for (int m = 0; m < DocumentsNumber; m++)
            {
                sumauthors += DA[m].Length;
            }

            double[] topics = new double[sumauthors];
            string[] authors = new string[sumauthors];
            int l = 0;
            for (int m = 0; m < DocumentsNumber; m++)
            {
                for (; l < DA[m].Length; l++)
                {
                    authors[l] = ThetaAT[m][l, TopicIndex].Author;
                    topics[l] = ThetaAT[m][l, TopicIndex].Prob;
                }
            }

            int i;
            double temptopics = 0;
            string tempauthorname = "";

            //build the initial heap
            for (i = (topics.Length - 1) / 2; i >= 0; i--)
                Adjust(i, topics.Length - 1, topics, authors);

            //swap root node and the last heap node
            for (i = topics.Length - 1; i >= 1; i--)
            {
                temptopics = topics[0];
                tempauthorname = authors[0];
                topics[0] = topics[i];
                authors[0] = authors[i];
                topics[i] = temptopics;
                authors[i] = tempauthorname;
                Adjust(0, i - 1, topics, authors);
            }
            Array.Reverse(authors);
            Array.Reverse(topics);
            string result = "Top " + MaxAuthor + " for this topic :" + Environment.NewLine;
            i = 0;
            while (MaxAuthor > i)
            {
                result += authors[i] + "\t" + topics[i] + Environment.NewLine;
                i++;
            }
            return result;
        }

        private static void Adjust(int root, int bottom, double[] topics, string[] authors)
        {
            double temptopics = topics[root];
            string tempauthorname = authors[root];

            int j = root * 2 + 1;

            while (j <= bottom)
            {
                //more children
                if (j < bottom)
                    if (topics[j] < topics[j + 1])
                        j = j + 1;

                //compare roots and the older children
                if (temptopics < topics[j])
                {
                    topics[root] = topics[j];
                    authors[root] = authors[j];
                    root = j;
                    j = 2 * root + 1;
                }
                else
                {
                    j = bottom + 1;
                }
            }
            topics[root] = temptopics;
            authors[root] = tempauthorname;
        }
    }
}
