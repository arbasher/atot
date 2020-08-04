using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using System.Collections;

public class ATGibbs
{
    double alpha, beta, investigatorParameter;
    int K, M, V, B, totalWords, totalUniquewords, iterations;
    string[] vocabArray;
    int[][] DW;
    int[][] WA;
   
    double[,] kLDTopics;
    double[,] nMITopics;
    double[,] kLDBTopicDocuments;
    string[][] DA;
    Random rand;
    Result[][] phiTW;
    Result[][,] thetaAT;
    Result[][] thetaDT;
    Result[][] PhiFT;
    DistanceMetric metric;

    public Result[][] PhiTW
    {
        get { return phiTW; }
    }

    public Result[][,] ThetaAT
    {
        get { return thetaAT; }
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


    public ATGibbs(double alpha, double beta, double investigatorParameter, int K, int[][] DW, string[][] DA, int[][] WA, Result[][] PhiFT, string[] vocabArray, int iterations)
    {
        this.alpha = alpha;
        this.beta = beta;
        this.investigatorParameter = investigatorParameter;
        this.K = K;
        this.DW = DW;
    
        this.DA = DA;
        this.WA = WA;
        this.vocabArray = vocabArray;
        B = PhiFT.Length;
        M = DW.Length;
        V = vocabArray.Length;
        this.iterations = iterations;
        rand = new Random();
        this.PhiFT = PhiFT;
        metric = new DistanceMetric(M, B, beta, K,V);
    }

    public ATGibbs(double alpha, double beta, int K, int[][] DW, string[][] DA, int[][] WA,string[] vocabArray, int iterations)
    {
        this.alpha = alpha;
        this.beta = beta;
        this.K = K;
        this.DW = DW;
        this.DA = DA;
        this.WA = WA;
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

        int[][] authorassign = new int[M][];
        int[,] nmk = new int[M, K];
        int[] nm = new int[M];
        int[,] nkv = new int[K, V];
        int[] nk = new int[K];
        int[][,] nat = new int[M][,]; //assigning the words to topics to an author for each document
        int[][] na = new int[M][]; // assigning the topics to authors for each document
        int[][] nbv = new int[B][];
        int[] nb = new int[B];
        int[] disticntWordsDoc;
        int[][,] natt = new int[M][,];
        kLDTopics = new double[B, K];
        nMITopics = new double[B, K];
        kLDBTopicDocuments = new double[M, B];

        for (int b = 0; b < B; b++)
        {
            nbv[b] = new int[PhiFT[b].Length];
        }

        totalWords=metric.WordsCollection(DW);
        totalUniquewords=metric.UniqueWordsCollection(DW);
        disticntWordsDoc = new int[totalUniquewords];
        metric.MapWordsDoc(DW, ref disticntWordsDoc);


        int z;
        int a;

        for (int m = 0; m < M; m++)
        {

            int N = DW[m].Length;
            int A = DA[m].Length;
            zassign[m] = new int[N];

            authorassign[m] = new int[N];
            na[m] = new int[A];
            nat[m] = new int[A, K];
            nm[m] = N;


            for (int n = 0; n < N; n++)
            {
                z = (int)rand.Next(0, K);
                int v = DW[m][n];
                a = WA[m][n];

                nmk[m, z]++;
                nat[m][a, z]++;
                na[m][a]++;
                nkv[z, v]++;
                nk[z]++;

                authorassign[m][n] = a;
                zassign[m][n] = z;



                for (int b = 0; b < B; b++)
                {
                    int index = metric.ArrayIndexofPhiFT(PhiFT, b, v);
                    if (index != -1)
                    {
                        nbv[b][index]++;
                        nb[b]++;
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
                    int A = DA[m].Length;

                    for (int n = 0; n < N; n++)
                    {
                        z = zassign[m][n];
                        a = authorassign[m][n];
             
                        int v = DW[m][n];
                        int index = Array.IndexOf(disticntWordsDoc, v);
                        nmk[m, z]--;
                        nm[m]--;

                        na[m][a]--;
                        nat[m][a, z]--;

                        nkv[z, v]--;
                        nk[z]--;

                        SampleZ(nkv, nk, nat, na, m, n, DA[m].Length, ref z,index);

                        na[m][a]++;
                        nat[m][a, z]++;

                        nmk[m, z]++;
                        nm[m]++;

                        nkv[z, v]++;
                        nk[z]++;

                        authorassign[m][n] = a;
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
        metric.NormalizedMutualInformation(nkv, nk, nbv, nb, PhiFT, nMITopics,totalWords);

        phiTW = new Result[K][];

        thetaDT = new Result[M][];
        thetaAT = new Result[M][,];
 

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
            thetaAT[m] = new Result[DA[m].Length, K];
            thetaDT[m] = new Result[K];

            for (int k = 0; k < K; k++)
            {
                thetaDT[m][k] = new Result("Topic" + k, (Convert.ToDouble(nmk[m, k]) + alpha) / Convert.ToDouble(nm[m] + (K * alpha)));
            }
            Array.Sort(thetaDT[m]);

            for (int x = 0; x < DA[m].Length; x++)
            {
                for (int k = 0; k < K; k++)
                {
                    thetaAT[m][x, k] = new Result(DA[m][x].ToString(), "Topic" + k, Convert.ToDouble(nat[m][x, k] + alpha) / Convert.ToDouble(na[m][x] + (K * alpha)));
                }  
            }
        }

    }

    public void MCMCPerplexity()
    {
        int[][] zassign = new int[M][];

        int[][] authorassign = new int[M][];
        int[,] nmk = new int[M, K];
        int[] nm = new int[M];
        int[,] nkv = new int[K, V];
        int[] nk = new int[K];
        int[][,] nat = new int[M][,]; //assigning the words to topics to an author for each document
        int[][] na = new int[M][]; // assigning the topics to authors for each document
     
        int[] disticntWordsDoc;
        int[][,] natt = new int[M][,];
   
        totalWords = metric.WordsCollection(DW);
        totalUniquewords = metric.UniqueWordsCollection(DW);
        disticntWordsDoc = new int[totalUniquewords];
        metric.MapWordsDoc(DW, ref disticntWordsDoc);


        int z;
        int a;

        for (int m = 0; m < M; m++)
        {

            int N = DW[m].Length;
            int A = DA[m].Length;
            zassign[m] = new int[N];
            authorassign[m] = new int[N];
            na[m] = new int[A];
            nat[m] = new int[A, K];
            nm[m] = N;


            for (int n = 0; n < N; n++)
            {
                z = (int)rand.Next(0, K);
                int v = DW[m][n];
                a = WA[m][n];

                nmk[m, z]++;
                nat[m][a, z]++;
                na[m][a]++;
                nkv[z, v]++;
                nk[z]++;

                authorassign[m][n] = a;
                zassign[m][n] = z;

            }
        }

        Console.Write("Type (q) to quit or any charecter to redraw sample? Ans: ");
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
                    int A = DA[m].Length;

                    for (int n = 0; n < N; n++)
                    {
                        z = zassign[m][n];
                        a = authorassign[m][n];

                        int v = DW[m][n];
                        int index = Array.IndexOf(disticntWordsDoc, v);
                        nmk[m, z]--;
                        nm[m]--;

                        na[m][a]--;
                        nat[m][a, z]--;

                        nkv[z, v]--;
                        nk[z]--;

                        SampleZ(nkv, nk, nat, na, m, n, DA[m].Length, ref z, index);

                        na[m][a]++;
                        nat[m][a, z]++;

                        nmk[m, z]++;
                        nm[m]++;

                        nkv[z, v]++;
                        nk[z]++;

                        authorassign[m][n] = a;
                        zassign[m][n] = z;
                    }
                }
                i--;
            }

            Console.Write("Type (q) to quit or any charecter to redraw sample? Ans: ");
            answer = Console.ReadLine().ToLower();
        }

        thetaAT = new Result[M][,];
  
        for (int m = 0; m < M; m++)
        {
            thetaAT[m] = new Result[DA[m].Length, K];
           
            for (int x = 0; x < DA[m].Length; x++)
            {
                for (int k = 0; k < K; k++)
                {
                    thetaAT[m][x, k] = new Result(DA[m][x].ToString(), "Topic" + k, Convert.ToDouble(nat[m][x, k] + alpha) / Convert.ToDouble(na[m][x] + (K * alpha)));
                }
            }
        }

    }

    private void SampleZ(int[,] nkv, int[] nk, int[][,] nat, int[][] na, int ndocument, int nwords, int nauthors, ref int z, int distinct)
    {
        double[] totprob = new double[nauthors];
        double totprobauthor = 0;
        double[][] totauthortopicprob = new double[nauthors][];

        int v = DW[ndocument][nwords];

        for (int x = 0; x < nauthors; x++)
        {
            totauthortopicprob[x] = new double[K];
            for (int k = 0; k < K; k++)
            {
                totauthortopicprob[x][k] = (Convert.ToDouble(nkv[k, v] + beta) / Convert.ToDouble(nk[k] + (V * beta))) *
                       (Convert.ToDouble(nat[ndocument][x, k] + alpha) / Convert.ToDouble(na[ndocument][x] + (K * alpha)));
                   
                totprobauthor += totauthortopicprob[x][k];
            }
            totprob[x] = totprobauthor;
            totprobauthor = 0;
        }

        int maxauthor = Array.IndexOf(totprob, totprob.Max());

        for (int k = 1; k < K; k++)
        {
            totauthortopicprob[maxauthor][k] += totauthortopicprob[maxauthor][k - 1];
        }

        double u = (rand.NextDouble() * totprob[maxauthor]);

        for (z = 0; z < K; z++)
        {
            if (totauthortopicprob[maxauthor][z] > u)
            {
                break;
            }
            if (z + 1 == K)
                break;
        }
    }

}

