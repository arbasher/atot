using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Collections;

public class LDAGibbs
{
    double alpha, beta, investigatorParameter;
    int K, M, V, B, totalWords, totalUniquewords, iterations;
    string[] vocabArray;
    int[][] DW;
    Random rand;
    Result[][] phiTW;
    Result[][] thetaDT;
    Result[][] PhiFT;
    DistanceMetric metric;
    double[,] kLDTopics;
    double[,] nMITopics;
    double[,] kLDBTopicDocuments;

    public Result[][] PhiTW
    {
        get { return phiTW; }
    }

    public Result[][] ThetaDT
    {
        get { return thetaDT; }
    }

    public double[,] KLDTopics
    {
        get { return kLDTopics; }
    }

    public double[,] NMITopics
    {
        get { return nMITopics; }
    }

    public double[,] KLDBTopicDocuments
    {
        get { return kLDBTopicDocuments; }
    }

    public LDAGibbs(double alpha, double beta, double investigatorParameter, int K, int[][] DW, Result[][] PhiFT, string[] vocabArray, int iterations)
    {
        this.alpha = alpha;
        this.beta = beta;
        this.investigatorParameter = investigatorParameter;
        this.K = K;
        this.DW = DW;
        this.vocabArray = vocabArray;
        B = PhiFT.Length;
        M = DW.Length;
        V = vocabArray.Length;
        this.iterations = iterations;
        rand = new Random();
        this.PhiFT = PhiFT;
        metric = new DistanceMetric(M, B, beta, K, V);
    }

    public LDAGibbs(double alpha, double beta, int K, int[][] DW, string[] vocabArray, int iterations)
    {
        this.alpha = alpha;
        this.beta = beta;
        this.K = K;
        this.DW = DW;
        this.vocabArray = vocabArray;
        M = DW.Length;
        V = vocabArray.Length;
        this.iterations = iterations;
        rand = new Random();
        metric = new DistanceMetric(M, beta, K, V);
    }

    public void MCMC()
    {
        int[][] zassign = new int[M][];
        int[,] nmk = new int[M, K];
        int[] nm = new int[M];
        int[,] nkv = new int[K, V];
        int[] nk = new int[K];
        int[][] nbv = new int[B][];
        int[] nb = new int[B];
        int[] disticntWordsDoc;
        kLDTopics = new double[B, K];
        nMITopics = new double[B, K];
        kLDBTopicDocuments = new double[M, B];

        for (int b = 0; b < B; b++)
        {
            nbv[b] = new int[PhiFT[b].Length];
        }

        totalWords = metric.WordsCollection(DW);
        totalUniquewords = metric.UniqueWordsCollection(DW);
        disticntWordsDoc = new int[totalUniquewords];
        metric.MapWordsDoc(DW, ref disticntWordsDoc);

        int z;

        for (int m = 0; m < M; m++)
        {
            int N = DW[m].Length;
            zassign[m] = new int[N];
            nm[m] = N;

            for (int n = 0; n < N; n++)
            {
                z = (int)rand.Next(0, K);
                nmk[m, z]++;

                int v = DW[m][n];
                nkv[z, v]++;
                nk[z]++;
                zassign[m][n] = z;

                for (int b = 0; b < B; b++)
                {
                    int index =metric.ArrayIndexofPhiFT( PhiFT,b,v);
                    if (index != -1)
                    {
                        nbv[b][index]++;
                        nb[b]++;
                    }

                }
            }
        }

        //double minimumSupportKLDsBT= metric.KLdivergenceDocumentsBackgroundTopic(nbv, DW, PhiFT, KLDBTopicDocuments, disticntWordsDoc, totalWords, investigatorParameter);

        //Console.WriteLine("The minimum support for the KL divergence between the document model and the given background topic model is {0}", minimumSupportKLDsBT);

        int indicator = B;
        //for (int d = 0; d < M; d++)
        //{
        //    for (int b = 0; b < B; b++)
        //    {
        //        Console.WriteLine("The KL divergence between the document model and the given background topic model {0} for the document {1} is ({2})", b, d, kLDBTopicDocuments[d, b]);
        //    }
        //}

        //Console.Write("Type (q) to quit or any charecter to continue? Ans: ");
        //string answer = Console.ReadLine().ToLower();

        //while (answer != "q")
        //{
            DateTime startTime = DateTime.Now;

            int loops = iterations;

            while (loops > 0)
            {
                Console.WriteLine("Iteration: {0}", loops);
                for (int m = 0; m < M; m++)
                {
                    int N = DW[m].Length;
                    for (int n = 0; n < N; n++)
                    {
                        z = zassign[m][n];

                        int v = DW[m][n];
                        int index = Array.IndexOf(disticntWordsDoc, v);
                        nmk[m, z]--;
                        nm[m]--;

                        nkv[z, v]--;
                        nk[z]--;

                        z = SampleZ(nkv, nk, nmk, nm, m, n, index);

                        nmk[m, z]++;
                        nm[m]++;

                        nkv[z, v]++;
                        nk[z]++;

                        zassign[m][n] = z;
                    }
                }
                loops--;
            }
            DateTime finishTime = DateTime.Now;

            Array.Clear(kLDTopics, 0, kLDTopics.Length);

            double minimumSupport = metric.KLdivergenceTopicsBackgroundTopic(nkv, nk, nbv, PhiFT, kLDTopics, 0, totalUniquewords, investigatorParameter);

            indicator = nb.Length;
            double distanceTopic = 0;

            for (int d = 0; d < M; d++)
            {
                for (int b = 0; b < B; b++)
                {
                    for (int k = 0; k < K; k++)
                    {
                        if (minimumSupport > kLDTopics[b, k] && kLDTopics[b, k] != -1)
                        {
                            distanceTopic = kLDTopics[b, k];
                            indicator--;
                            break;
                        }
                    }
                }
            }
            if (indicator != 0)
            {
                Console.WriteLine("The KL divergence did not find any topic that supports minimum threshold");
            }
            else
            {
                Console.WriteLine("The KL divergence found a topic with {0} score", distanceTopic);
            }
        //    Console.Write("Type (q) to quit or any charecter to continue? Ans: ");
        //    answer = Console.ReadLine().ToLower();
        //}
           
            Console.WriteLine("Started at: " + startTime + " Finished at: " + finishTime);

        metric.NormalizedMutualInformation(nkv, nk, nbv, nb, PhiFT, nMITopics, totalWords);

        phiTW = new Result[K][];

        thetaDT = new Result[M][];

        for (int k = 0; k < K; k++)
        {
            phiTW[k] = new Result[V];
            for (int v = 0; v < V; v++)
            {
                phiTW[k][v] = new Result(vocabArray[v], (Convert.ToDouble(nkv[k, v]) + beta) / Convert.ToDouble(nk[k] + (V * beta)));
            }

            Array.Sort(phiTW[k]);
        }

        for (int m = 0; m < M; m++)
        {
            thetaDT[m] = new Result[K];

            for (int k = 0; k < K; k++)
            {
                thetaDT[m][k] = new Result("Topic" + k, (Convert.ToDouble(nmk[m, k]) + alpha) / Convert.ToDouble(nm[m] + (K * alpha)));
            }
            Array.Sort(thetaDT[m]);
        }
    }

    public void MCMCPerplexity()
    {
        int[][] zassign = new int[M][];
        int[,] nmk = new int[M, K];
        int[] nm = new int[M];
        int[,] nkv = new int[K, V];
        int[] nk = new int[K];
        int[] disticntWordsDoc;
  
        totalWords = metric.WordsCollection(DW);
        totalUniquewords = metric.UniqueWordsCollection(DW);
        disticntWordsDoc = new int[totalUniquewords];
        metric.MapWordsDoc(DW, ref disticntWordsDoc);

        int z;

        for (int m = 0; m < M; m++)
        {
            int N = DW[m].Length;
            zassign[m] = new int[N];
            nm[m] = N;

            for (int n = 0; n < N; n++)
            {
                z = (int)rand.Next(0, K);
                nmk[m, z]++;
                int v = DW[m][n];
                nkv[z, v]++;
                nk[z]++;
                zassign[m][n] = z;
            }
        }

        Console.Write("Type (q) to quit or any charecter to redraw the sample? Ans: ");
        string answer = Console.ReadLine().ToLower();

        while (answer != "q")
        {
            int i = iterations;

            while (i > 0)
            {
                Console.WriteLine("Iteration: {0}", i);
                for (int m = 0; m < M; m++)
                {
                    int N = DW[m].Length;
                    for (int n = 0; n < N; n++)
                    {
                        z = zassign[m][n];

                        int v = DW[m][n];
                        int index = Array.IndexOf(disticntWordsDoc, v);
                        nmk[m, z]--;
                        nm[m]--;

                        nkv[z, v]--;
                        nk[z]--;

                        z = SampleZ(nkv, nk, nmk, nm, m, n, index);

                        nmk[m, z]++;
                        nm[m]++;

                        nkv[z, v]++;
                        nk[z]++;

                        zassign[m][n] = z;
                    }
                }
                i--;
            }
            Console.Write("Type (q) to quit or any charecter to redraw the sample? Ans: ");
            answer = Console.ReadLine().ToLower();
        }

        thetaDT = new Result[M][];

        for (int m = 0; m < M; m++)
        {
            thetaDT[m] = new Result[K];

            for (int k = 0; k < K; k++)
            {
                thetaDT[m][k] = new Result("Topic" + k, (Convert.ToDouble(nmk[m, k]) + alpha) / Convert.ToDouble(nm[m] + (K * alpha)));
            }
            Array.Sort(thetaDT[m]);
        }
    }

    private int SampleZ(int[,] nkv, int[] nk, int[,] nmk, int[] nm, int m, int n, int distinct)
    {
        double[] p = new double[K];
        for (int k = 0; k < K; k++)
        {
            int v = DW[m][n];
            p[k] = ((Convert.ToDouble(nkv[k, v]) + beta) / Convert.ToDouble(nk[k] + (V * beta))) *
                    ((Convert.ToDouble(nmk[m, k]) + alpha) / Convert.ToDouble(nm[m] + (K * alpha)));
        }

        // cumulate multinomial parameters
        for (int k = 1; k < K; k++)
        {
            p[k] += p[k - 1];
        }

        // scaled sample because of unnormalized p[]
        double u = rand.NextDouble() * p[K - 1];
        int z;
        for (z = 0; z < K; z++)
        {
            if (p[z] > u)
            {
                break;
            }
            if (z + 1 == K)
                break;
        }
        return z;
    }

}