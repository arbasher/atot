using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

public class Parameters
    {
        private int topicsNumber;
        private int maxVocabNumber;
        private int iterations;
        private double alpha;
        private double beta;
        private double investigatorParameter;

        public Parameters(int K, int V,int iterations)
        {
            topicsNumber = K;
            maxVocabNumber = V;
            this.iterations = iterations;
            this.alpha = 50 / K;
            this.beta = 0.01;
            this.investigatorParameter = 0.5;
        }

        public Parameters(int K, int V, int iterations, double alpha, double beta, double investigatorParameter)
        {
            topicsNumber = K;
            maxVocabNumber = V;
            this.iterations = iterations;
            this.alpha = alpha;
            this.beta = beta;
            this.investigatorParameter = investigatorParameter;
        }

        public int TopicsNumber
        {
            get { return topicsNumber; }
        }

        public int MaxVocabNumber
        {
            get { return maxVocabNumber; }
        }

        public int Iterations
        {
            get { return iterations; }
        }

        public double Alpha
        {
            get { return alpha; }
        }

        public double Beta
        {
            get { return beta; }
        }

        public double InvestigatorParameter
        {
            get { return investigatorParameter; }
        }

    }