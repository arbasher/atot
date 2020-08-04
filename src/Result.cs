using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

public class Result : IComparable
{
    int index;
    double prob;
    string word;
    string topic;
    string author;

    public int Index
    {
        get { return index; }
    }

    public double Prob
    {
        get { return prob; }
    }

    public string Word
    {
        get { return word; }
    }

    public string Topic
    {
        get { return topic; }
    }

    public string Author
    {
        get { return author; }
    }

    public Result(double prob)
    {
        this.prob = prob;
    }

    public Result(string word, double prob)
    {
        this.word = word;
        this.prob = prob;
    }

    public Result(string topic, string word,int index)
    {
        this.topic = topic;
        this.word = word;
        this.index = index;
    }

    public Result(string author, string topic, double prob)
    {
        this.author = author;
        this.topic = topic;
        this.prob = prob;
    }

    public Result(string word, double prob, int index)
    {
        this.word = word;
        this.prob = prob;
        this.index = index;
    }

    public int CompareTo(Object obj)
    {
        Result r = obj as Result;
        if (prob > r.prob)
            return -1;
        else if (prob < r.prob)
            return 1;
        else
            return 0;
    }

}
