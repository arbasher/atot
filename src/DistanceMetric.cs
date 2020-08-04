using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Collections;


class DistanceMetric
{
    private int M;
    private int B;
    private double beta;
    private int K;
    private int V;
    double[][] tF;

    public double[][] TF
    {
        get { return tF; }
    }

    public DistanceMetric(int numberDocuments, int numberBackTopics, double beTa, int numberTopics, int vocabularyNumber)
    {
        M = numberDocuments;
        B = numberBackTopics;
        beta = beTa;
        K = numberTopics;
        V = vocabularyNumber;
    }

    public DistanceMetric(int numberDocuments, double beTa, int numberTopics, int vocabularyNumber)
    {
        M = numberDocuments;
        beta = beTa;
        K = numberTopics;
        V = vocabularyNumber;
    }

    public double KLdivergenceDocumentsBackgroundTopic(int[][] nbv, int[][] DW, Result[][] PhiFT, double[,] KLDBTopicDocuments, int[] uniQue,int totalWords, double investigatorParameter)
    {
        //double average = 0;
        //double jensenShannondivergence = 0;
        double minimumSupport = 0;

        for (int m = 0; m < M; m++)
        {
            for (int b = 0; b < B; b++)
            {
                //for (int v = 0; v < PhiFT[b].Length; v++)
                //{
                //    double p = (nbv[b][v] + beta) / (DW[m].Length + (uniQue.Length * beta));
                //    double q = PhiFT[b][v].Prob;
                //    //average = (p + q) / 2;
                //    //jensenShannondivergence = ((p * Math.Log10(p / average))  + (q * Math.Log10(q / average))) / 2;
                //    //KLDBTopicDocuments[m, b] += jensenShannondivergence;

                //    test1 += Math.Log(p / q);
                //    test2 += Math.Log(q / p);

                //    KLDBTopicDocuments[m, b] += (((p * Math.Log(p / q)) + (q * Math.Log(q / p))) / 2);
                //}

                for (int v = 0; v < V; v++)
                {
                    int totalWordsPerDocument = 0;
                    for (int i = 0; i < DW[m].Length; i++)
                        if (DW[m][i] == v)
                            totalWordsPerDocument++;

                    double d = (totalWordsPerDocument + beta) / (totalWords + (V * beta));

                    double bt = 0;

                    int arrayIndexofPhiFT = ArrayIndexofPhiFT(PhiFT, b, v);
                    if (arrayIndexofPhiFT != -1)
                        bt = PhiFT[b][arrayIndexofPhiFT].Prob;
                    else
                        continue;
                    //average = (p + q) / 2;
                    //jensenShannondivergence = ((p * Math.Log10(p / average))  + (q * Math.Log10(q / average))) / 2;
                    //KLDBTopicDocuments[m, b] += jensenShannondivergence;

                    minimumSupport += bt * Math.Log(bt / d);

                    KLDBTopicDocuments[m, b] += (((d * Math.Log(d / bt)) + (bt * Math.Log(bt / d))) / 2);
                }
            }
        }

       //NormalizedKLDivergence(KLDBTopicDocuments);
       return minimumSupport = (minimumSupport/M) *investigatorParameter;
    }

    private void NormalizedKLDivergence(double[,] KLDBTopicDocuments)
    {
        double total = 0;
        double average = 0;
        double klsqrt = 0;
        double standard = 0;

        for (int d = 0; d < M; d++)
        {
            for (int b = 0; b < B; b++)
            {
                klsqrt += Math.Pow(KLDBTopicDocuments[d, b],2);
                total += KLDBTopicDocuments[d, b];
            }
        }
       
        average =total / M;
        standard = Math.Sqrt((klsqrt/M)-Math.Pow(2,average));
        
        for (int d = 0; d < M; d++)
        {
            for (int b = 0; b < B; b++)
            {
                KLDBTopicDocuments[d, b] = (KLDBTopicDocuments[d, b] - average) / standard;
                if (KLDBTopicDocuments[d, b] < 0)
                    KLDBTopicDocuments[d, b] = KLDBTopicDocuments[d, b] * -1;
            }
        }
    }

    public double KLdivergenceTopicsBackgroundTopic(int[,] nkv, int[] nk, int[][] nbv, Result[][] PhiFT, double[,] KLDsum, int index, int totalUniquewords, double investigatorParameter)
    {
        //double average = 0;
        //double jensenShannondivergence = 0;
        double minimumSupport = 0;

        for (int b = 0; b < B; b++)
        {
            for (int k = 0; k < K; k++)
            {
                //int relTopics = 0;
                //for (int w = 0; w < nbv[b].Length; w++)
                //{
                //    relTopics += nkv[k, PhiFT[b][w].Index];
                //}
                //if (relTopics != 0)
                //{
                //    for (int v = 0; v < nbv[b].Length; v++)
                //    {
                //        double p = (nkv[k, PhiFT[b][v].Index] + beta) / (nk[k] + (totalUniquewords * beta));
                //        double q = PhiFT[b][v].Prob;
                //        //average = (p + q) / 2;
                //        //jensenShannondivergence = ((p * Math.Log10(p / average)) + (q * Math.Log10(q / average))) / 2;
                //        //KLDsum[b, k] += jensenShannondivergence;
                //        test1 += Math.Log(p / q);
                //        test2 += Math.Log(q / p);
                //        KLDsum[b, k] += ((p * Math.Log(p / q)) + (q * Math.Log(q / p))) / 2;
                //    }
                //}
                //else
                //    KLDsum[b, k] = -1;

                for (int v = 0; v < V; v++)
                {
                    double t = 0;
                    double bt = 0;

                    if (nkv[k, v] != -1)
                        t = (nkv[k, v] + beta) / (nk[k] + (V * beta));

                    int arrayIndexofPhiFT = ArrayIndexofPhiFT(PhiFT, b, v);
                    if (arrayIndexofPhiFT != -1)
                        bt = PhiFT[b][arrayIndexofPhiFT].Prob;
                    else
                        continue;

                    minimumSupport += bt * Math.Log(bt / t);

                    KLDsum[b, k] += ((t * Math.Log(t / bt)) + (bt * Math.Log(bt / t))) / 2;
                }
            }
        }
      return minimumSupport = (minimumSupport/K) * investigatorParameter;
    }

    public void NormalizedMutualInformation(int[,] nkv, int[] nk, int[][] nbv, int[] nb, Result[][] PhiFT, double[,] nMITopics, int totalWords)
    {
        for (int b = 0; b < B; b++)
        {
            for (int k = 0; k < K; k++)
            {
                double relTopics = 0;
                for (int w = 0; w < nbv[b].Length; w++)
                {
                    relTopics += nkv[k, PhiFT[b][w].Index];
                }
                if (relTopics != 0)
                {
                    //double totalWords = nb[b] + nk[k] - relTopics;

                    //double MI = (relTopics / (totalWords) * Math.Log(((relTopics * totalWords) / (nk[k] * nb[b])), 2));
                    //MI += ((nb[b] - relTopics) / (totalWords) * Math.Log((((nb[b] - relTopics) * totalWords) / ((nb[b] - relTopics) * nb[b])), 2));
                    //MI += ((nk[k] - relTopics) / (totalWords) * Math.Log((((nk[k] - relTopics) * totalWords) / (((nk[k] - relTopics) * nk[k]))), 2));

                    double rest = totalWords - nk[k] - nb[b] + relTopics;

                    double MI = (relTopics / (totalWords) * Math.Log(((relTopics * totalWords) / (nk[k] * nb[b])), 2));
                    MI += ((nb[b] - relTopics) / (totalWords) * Math.Log((((nb[b] - relTopics) * totalWords) / ((rest + nb[b] - relTopics) * nb[b])), 2));
                    MI += ((nk[k] - relTopics) / (totalWords) * Math.Log((((nk[k] - relTopics) * totalWords) / (((rest + nk[k] - relTopics) * nk[k]))), 2));
                    MI += ((rest) / (totalWords) * Math.Log((((rest) * totalWords) / (((rest + nb[b] - relTopics) * (rest + nk[k] - relTopics)))), 2));

                    double Hk = -1 * Convert.ToDouble((double)nk[k] / (totalWords)) * Math.Log((((double)nk[k]) / (totalWords)), 2);
                    double Hb = -1 * Convert.ToDouble((double)nb[b] / (totalWords)) * Math.Log((((double)nb[b]) / (totalWords)), 2);

                    nMITopics[b, k] = (MI / ((Hk + Hb) / 2)) * 10;
                }
                else
                    nMITopics[b, k] = -1;
            }
        }
    }

    public double[] EstimateProbabilities(int[] probWT, int[][] tassign, int[][] DW, int m)
    {
        double[] prob;
        prob = new double[DW[m].Length];
        for (int n = 0; n < DW[m].Length; n++)
        {
            prob[n] = EstimateWordsProbabilities(DW, m, n, probWT[tassign[m][n]]);
        }

        return prob;
    }

    public double EstimateWordsProbabilities(int[][] DW, int m, int n, int total)
    {
        int nwords = 0;
        for (int i = 0; i < DW[m].Length; i++)
        {
            if (DW[m][n] == DW[m][i])
                nwords++;
        }
        return (nwords + beta) / (total + (DW[m].Length * beta));
    }

    public int WordsCollection(int[][] DW)
    {
        int totalWords = 0;
        for (int d = 0; d < DW.Length; d++)
            totalWords += DW[d].Length;
        return totalWords;
    }

    public void TermFrequency(Result[][] PhiFT, int[][] DW, int d)
    {
        tF = new double[B][];
        for (int t = 0; t < B; t++)
        {
            int index = 0;
            tF[t] = new double[PhiFT[t].Length];

            for (int w = 0; w < DW[d].Length; w++)
            {
                index = ArrayIndexofPhiFT(PhiFT, t, w);

                if (index != -1)
                {
                    tF[t][index]++;
                }
            }

            for (int i = 0; i < PhiFT[t].Length; i++)
            {
                tF[t][i] = (tF[t][i] + beta) / (DW[d].Length + (DW[d].Distinct().Count() * beta));
            }
        }
    }

    public void MapWordsDoc(int[][] DW, ref int[] disticntWordsDoc)
    {
        int index = 0;

        for (int m = 0; m < M; m++)
        {
            for (int n = 0; n < DW[m].Length; n++)
            {
                if (!disticntWordsDoc.Contains(DW[m][n]))
                {
                    disticntWordsDoc[index] = DW[m][n];
                    index++;
                }
            }
        }
    }

    public int UniqueWordsCollection(int[][] DW)
    {
        ArrayList uniquewords = new ArrayList();

        for (int d = 0; d < DW.Length; d++)
            for (int n = 0; n < DW[d].Length; n++)
                if (!uniquewords.Contains(DW[d][n]))
                    uniquewords.Add(DW[d][n]);
        return uniquewords.Count;
    }

    public int ArrayIndexofPhiFT(Result[][] BT, int backTopic, int v)
    {
        for (int i = 0; i < BT[backTopic].Length; i++)
        {
            if (BT[backTopic][i].Index == v)
                return i;
        }

        return -1;
    }

}

