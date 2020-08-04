using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Collections;

public class LDATOTGibbs
{
    double alpha, beta, investigatorParameter;
    int K, M, V, B, totalWords, totalUniquewords, iterations;
    string[] vocabArray;
    int[][] DW;
    double[] DT;
    double[][] WT;
    Random rand;
    Result[][] phiTW;
    Result[][] thetaDT;
    Result[][] PhiFT;
    Result[][] betaTOT;
    Result[][] betaFTOT;
    Result[][] dOT;
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

    public Result[][] BetaTOT
    {
        get { return betaTOT; }
    }

    public Result[][] BetaFTOT
    {
        get { return betaFTOT; }
    }

    public Result[][] DOT
    {
        get { return dOT; }
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

    public LDATOTGibbs(double alpha, double beta, double investigatorParameter, int K, int[][] DW, double[] DT, double[][] WT, Result[][] PhiFT, string[] vocabArray, int iterations)
    {
        this.alpha = alpha;
        this.beta = beta;
        this.investigatorParameter = investigatorParameter;
        this.K = K;
        this.DW = DW;
        this.DT = DT;
        this.WT = WT;
        this.vocabArray = vocabArray;
        B = PhiFT.Length;
        M = DW.Length;
        V = vocabArray.Length;
        this.iterations = iterations;
        rand = new Random();
        this.PhiFT = PhiFT;
        metric = new DistanceMetric(M, B, beta, K, V);
    }

    public LDATOTGibbs(double alpha, double beta,int K, int[][] DW, double[] DT, double[][] WT, string[] vocabArray, int iterations)
    {
        this.alpha = alpha;
        this.beta = beta;
        this.K = K;
        this.DW = DW;
        this.DT = DT;
        this.WT = WT;
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
        int[][] tassign = new int[M][];
        int[,] nmk = new int[M, K];
        int[] nm = new int[M];
        int[,] nkv = new int[K, V];
        int[] nk = new int[K];
        int[][,] nttv = new int[K][,];
        int[,] ntt = new int[K, DT.Length];
        int[,] nftt = new int[B, DT.Length];
        int[,] ndt = new int[M, DT.Length];
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

        for (int i = 0; i < K; i++)
        {
            nttv[i] = new int[DT.Length, totalUniquewords];
        }

        int z;
        int t;

        for (int m = 0; m < M; m++)
        {
            int N = DW[m].Length;
            zassign[m] = new int[N];
            tassign[m] = new int[N];
            nm[m] = N;

            for (int n = 0; n < N; n++)
            {
                z = (int)rand.Next(0, K);
                nmk[m, z]++;

                int v = DW[m][n];
                nkv[z, v]++;
                nk[z]++;
                zassign[m][n] = z;

                for (t = 0; t < DT.Length; t++)
                {
                    if (WT[m][n] <= DT[t])
                    {
                        nttv[z][t, Array.IndexOf(disticntWordsDoc, v)]++;
                        ntt[z, t]++;
                        tassign[m][n] = t;
                        ndt[m, t]++;
                        for (int b = 0; b < B; b++)
                        {
                            int index = metric.ArrayIndexofPhiFT(PhiFT, b, v);
                            if (index != -1)
                            {
                                nbv[b][index]++;
                                nb[b]++;
                                nftt[b, t]++;
                            }

                        }
                        break;
                    }
                    else if (DT.Length == t + 1)
                    {
                        nttv[z][0, Array.IndexOf(disticntWordsDoc, v)]++;
                        ntt[z, 0]++;
                        //add
                        ndt[m, 0]++;
                        tassign[m][n] = 0;
                        for (int b = 0; b < B; b++)
                        {
                            int index = metric.ArrayIndexofPhiFT(PhiFT, b, v);
                            if (index != -1)
                            {
                                nbv[b][index]++;
                                nb[b]++;
                                nftt[b, 0]++;
                            }

                        }
                        break;
                    }
                }
            }
        }

        //double minimumSupportKLDsBT = metric.KLdivergenceDocumentsBackgroundTopic(nbv, DW, PhiFT, KLDBTopicDocuments, disticntWordsDoc, totalWords, investigatorParameter);

        //Console.WriteLine("The minimum support for the KL divergence between the document model and the given background topic model is {0}", minimumSupportKLDsBT);

        int indicator = B;
        //for (int d = 0; d < M; d++)
        //{
        //    for (int b = 0; b < B; b++)
        //    {
        //        Console.WriteLine("The KL divergence between the document model and the given topic-words model {0} distribution is ({1})", b, kLDBTopicDocuments[d, b]);
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
                        t = tassign[m][n];
                        int v = DW[m][n];
                        int index = Array.IndexOf(disticntWordsDoc, v);
                        nmk[m, z]--;
                        nm[m]--;

                        nkv[z, v]--;
                        nk[z]--;

                        if (ntt[z, t] != 0)
                            ntt[z, t]--;
                        nttv[z][t, index]--;

                        z = SampleZ(nkv, nk, nmk, nm, nttv, ntt, m, n, t, index);

                        nttv[z][t, index]++;
                        ntt[z, t]++;

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
        betaTOT = new Result[K][];
        betaFTOT = new Result[B][];
        dOT=new Result[M][];
      
        thetaDT = new Result[M][];

        for (int b = 0; b < B; b++)
        {
            betaFTOT[b] = new Result[DT.Length];
            for (int s = 0; s < DT.Length; s++)
            {
                betaFTOT[b][s] = new Result((Convert.ToDouble(nftt[b, s] + beta) / (nb[b] + (V * beta))));
            }
        }

        for (int k = 0; k < K; k++)
        {
            phiTW[k] = new Result[V];
            betaTOT[k] = new Result[DT.Length];

            for (int v = 0; v < V; v++)
            {
                phiTW[k][v] = new Result(vocabArray[v], (Convert.ToDouble(nkv[k, v]) + beta) / Convert.ToDouble(nk[k] + (V * beta)));
            }
            for (int s = 0; s < DT.Length; s++)
                betaTOT[k][s] = new Result("Topic" + k, (Convert.ToDouble(ntt[k, s] + beta) / Convert.ToDouble(nk[k] + (V * beta))));

            Array.Sort(phiTW[k]);
        }

        for (int m = 0; m < M; m++)
        {
            thetaDT[m] = new Result[K];
            dOT[m] = new Result[DT.Length];

            for (int k = 0; k < K; k++)
            {
                thetaDT[m][k] = new Result("Topic" + k, (Convert.ToDouble(nmk[m, k]) + alpha) / Convert.ToDouble(nm[m] + (K * alpha)));
            }
            Array.Sort(thetaDT[m]);

            for (int s = 0; s < DT.Length; s++)
                dOT[m][s] = new Result("document" + m, (Convert.ToDouble(ndt[m, s] + beta) / Convert.ToDouble(nm[m] + (V * beta))));

        }
    }

    public void MCMCPerplexity()
    {
        int[][] zassign = new int[M][];
        int[][] tassign = new int[M][];
        int[,] nmk = new int[M, K];
        int[] nm = new int[M];
        int[,] nkv = new int[K, V];
        int[] nk = new int[K];
        int[][,] nttv = new int[K][,];
        int[,] ntt = new int[K, DT.Length];
        int[,] ndt = new int[M, DT.Length];
        int[] disticntWordsDoc;

        totalWords = metric.WordsCollection(DW);
        totalUniquewords = metric.UniqueWordsCollection(DW);
        disticntWordsDoc = new int[totalUniquewords];
        metric.MapWordsDoc(DW, ref disticntWordsDoc);

        for (int i = 0; i < K; i++)
        {
            nttv[i] = new int[DT.Length, totalUniquewords];
        }

        int z;
        int t;

        for (int m = 0; m < M; m++)
        {
            int N = DW[m].Length;
            zassign[m] = new int[N];
            tassign[m] = new int[N];
            nm[m] = N;

            for (int n = 0; n < N; n++)
            {
                z = (int)rand.Next(0, K);
                int v = DW[m][n];
                nmk[m, z]++;
                nkv[z, v]++;
                nk[z]++;
                zassign[m][n] = z;

                for (t = 0; t < DT.Length; t++)
                {
                    if (WT[m][n] <= DT[t])
                    {
                        nttv[z][t, Array.IndexOf(disticntWordsDoc, v)]++;
                        ntt[z, t]++;
                        tassign[m][n] = t;
                        ndt[m, t]++;
                        break;
                    }
                    else if (DT.Length == t + 1)
                    {
                        nttv[z][0, Array.IndexOf(disticntWordsDoc, v)]++;
                        ntt[z, 0]++;
                        //add
                        ndt[m, 0]++;
                        tassign[m][n] = 0;
                        break;
                    }
                }
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
                        t = tassign[m][n];
                        int v = DW[m][n];
                        int index = Array.IndexOf(disticntWordsDoc, v);
                        nmk[m, z]--;
                        nm[m]--;

                        nkv[z, v]--;
                        nk[z]--;

                        if (ntt[z, t] != 0)
                            ntt[z, t]--;
                        nttv[z][t, index]--;

                        z = SampleZ(nkv, nk, nmk, nm, nttv, ntt, m, n, t, index);

                        nttv[z][t, index]++;
                        ntt[z, t]++;

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

    private int SampleZ(int[,] nkv, int[] nk, int[,] nmk, int[] nm, int[][,] nttv, int[,] ntt, int m, int n, int s, int distinct)
    {
        double[] p = new double[K];
        for (int k = 0; k < K; k++)
        {
            int v = DW[m][n];
            p[k] = ((Convert.ToDouble(nkv[k, v]) + beta) / Convert.ToDouble(nk[k] + (V * beta))) *
                    ((Convert.ToDouble(nmk[m, k]) + alpha) / Convert.ToDouble(nm[m] + (K * alpha))) *
                    ((Convert.ToDouble(nttv[k][s, distinct]) + beta) / Convert.ToDouble(ntt[k, s] + (V * beta)));
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