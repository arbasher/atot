using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;

public class TextPath
{
    private string directorypath;
    private string indexpath;
    private string vocabpath;
    private string bgtopicspath;
    private string testfilespath;

    public TextPath()
    { 
    directorypath = @"..\..\..\..\Text Corpus\";
    indexpath = directorypath + @"\Index";
    vocabpath = indexpath + @"\vocab.txt";
    bgtopicspath = directorypath + @"\BgTopics";
    testfilespath = directorypath + @"\TestFiles";
    }

    public string Directorypath
    {
        get
        {
            return directorypath;
        }
    }

    public string Indexpath
    {
        get
        {
            return indexpath;
        }
    }

    public string Vocabpath
    {
        get
        {
            return vocabpath;
        }
    }

    public string Bgtopicspath
    {
        get
        {
            return bgtopicspath;
        }
    }

    public string Testfilespath
    {
        get
        {
            return testfilespath;
        }
    }

}

