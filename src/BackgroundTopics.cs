using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;


class BackgroundTopics
{
    Result[][] phiFT;
    Result[] similarityBTDW;
    Result[][] docsSimilarity;
    private int[][] DW;
    private string[] vocabArray;
    double beta;
    int B, M, V, totalUniquewords, iterations;
    Random rand;
    DistanceMetric metric;
    private int totalWords;

    public Result[][] PhiFT
    {
        get { return phiFT; }
    }

    public Result[] SimilarityBTDW
    {
        get { return similarityBTDW; }
    }

    public Result[][] DocsSimilarity
    {
        get { return docsSimilarity; }
    }

    public BackgroundTopics(double beta, int K, int[][] DW, string[] vocabArray, int iterations)
    {
        this.beta = beta;
        this.B = K;
        this.DW = DW;
        this.vocabArray = vocabArray;
        M = DW.Length;
        V = vocabArray.Length;
        this.iterations = iterations;
        rand = new Random();
        metric = new DistanceMetric(M, K, beta, -1,V);
    }

    public void BackGroundTopicsMCMC()
    {
        int[][] zassign = new int[M][];
        int[,] ndw = new int[M, V];
        int[,] nbv = new int[B, V];
        int[] nb = new int[B];
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

            for (int n = 0; n < N; n++)
            {
                z = (int)rand.Next(0, B);

                int v = DW[m][n];

                ndw[m, v]++;
                nbv[z, v]++;
                nb[z]++;

                zassign[m][n] = z;

            }
        }


        Console.Write("Type (q) to quit or any charecter to continue? Ans: ");
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

                        nbv[z, v]--;
                        nb[z]--;

                        z = SampleZ(nbv, nb, m, n, index);

                        nbv[z, v]++;
                        nb[z]++;

                        zassign[m][n] = z;
                    }
                }
                i--;
            }

            Console.Write("Type (q) to quit or any character to continue? Ans: ");
            answer = Console.ReadLine().ToLower();
        }
        phiFT = new Result[B][];

        for (int b = 0; b < B; b++)
        {
            phiFT[b] = new Result[V];
            for (int v = 0; v < V; v++)
            {
                phiFT[b][v] = new Result(vocabArray[v], (nbv[b, v] + beta) / (nb[b] + (V * beta)));
            }

        }

        Console.WriteLine("Search Cosine Similarity between BT and Docs? yes<y> or any character to cancel");
        answer=Console.ReadLine();
        if (answer == "y")
        {
            int j = 0;
            Result[][] phiDW = new Result[M][];
            Result[][] phiDocs=new Result[4][];
            similarityBTDW = new Result[M];

            for (int m = 0; m < M; m++)
            {
              phiDW[m] = new Result[V];

              for (int v = 0; v < V; v++)
              {
                  phiDW[m][v] = new Result(vocabArray[v], (ndw[m, v] + beta) / (totalWords + (V * beta)));
              }

              if (m == 135 || m == 137 || m == 231 || m == 244)
              {
                  phiDocs[j] = phiDW[m];
                  j++;
              }

             similarityBTDW[m]= new Result(GetCosineSimilarity(phiFT[0],phiDW[m]));
            }

            Console.WriteLine("Search Cosine Similarity between Docs? yes<y> or any character to cancel");

            answer = Console.ReadLine();
            if (answer == "y")
            {
                docsSimilarity = new Result[4][];

                for (int i = 0; i < 4; i++)
                {
                    docsSimilarity[i] = new Result[4];

                    for (j = 0; j < 4; j++)
                    {
                        docsSimilarity[i][j] = new Result(j.ToString(), GetCosineSimilarity(phiDocs[i], phiDocs[j]));
                    }
                }
            }
        }

        
    }

    private int SampleZ(int[,] nbv, int[] nb, int m, int n, int index)
    {
        double[] p = new double[B];

        for (int k = 0; k < B; k++)
        {
            int v = DW[m][n];
            p[k] = (nbv[k, v] + beta) / (nb[k] + (V * beta));
        }

        // cumulate multinomial parameters
        for (int k = 1; k < B; k++)
        {
            p[k] += p[k - 1];
        }

        // scaled sample because of unnormalized p[]
        double u = rand.NextDouble() * p[B - 1];
        int z;
        for (z = 0; z < B; z++)
        {
            if (p[z] > u)
            {
                break;
            }
            if (z + 1 == B)
                break;
        }
        return z;
    }

    public double GetCosineSimilarity(Result[] V1, Result[] V2)
    {
        double dot = 0.0d;
        double mag1 = 0.0d;
        double mag2 = 0.0d;

        for (int v = 0; v < V; v++)
        {
            dot += V1[v].Prob * V2[v].Prob;
            mag1 += Math.Pow(V1[v].Prob, 2);
            mag2 += Math.Pow(V2[v].Prob, 2);
        }

        return dot / (Math.Sqrt(mag1) * Math.Sqrt(mag2));
    }
}

