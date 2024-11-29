# PDF-to-Tree

## Introduction
In many PDF documents, the reading order of text blocks is missing, which can hinder machine understanding of the document's content.
Existing works try to extract one universal reading order for a PDF file.
However, applications, like Retrieval Augmented Generation (RAG), require breaking long articles into sections, subsections and table cells for better indexing.
For this reason, this paper introduces a new task and dataset, PDF-to-Tree, which organizes the text blocks of a PDF into a tree structure.

**PDF-to-Tree: Parsing PDF Text Blocks into a Tree**<br>
Yue Zhang, Zhihao Zhang, Wenbin Lai, Chong Zhang, Tao Gui, Qi Zhang, Xuanjing Huang<br>
Findings of the Association for Computational Linguistics: EMNLP 2024

## Dataset
You can download the dataset from here, [[Tree](https://1drv.ms/u/s!Aheo9Mq3aL6Vg-81oAJpGvCWhXG0Eg?e=r0EcVX)],[[Image](https://1drv.ms/u/s!Aheo9Mq3aL6Vg-8yWW-CV93E-rUqPg?e=UWd7Sf)].

Then you should unzip the tree and image files, and preprocess the data by running the following command.

```
python3 preprocess.py -c configs/example.json -i <path/to/tree> -img <path/to/image> -o <path/to/output>
```

Or you can use the preprocessed data from [[Here](https://1drv.ms/u/s!Aheo9Mq3aL6Vg-8z5s97WnH9UGdtJg?e=EicAJj)].

## Run Training

You can use the following command to run training.

```
python3 train.py -c config/example.json -d <path/to/data> -w <path/to/workdir>
```

You can check the results in the `<path/to/workdir>` directory. 
The predicted tree is saved as `test-parser_preds.json`.
You can also find the metrics in `test-parser_metrics.json`.

## Citation
```
@inproceedings{zhang-etal-2024-pdf,
    title = "{PDF}-to-Tree: Parsing {PDF} Text Blocks into a Tree",
    author = "Zhang, Yue  and
      Zhang, Zhihao  and
      Lai, Wenbin  and
      Zhang, Chong  and
      Gui, Tao  and
      Zhang, Qi  and
      Huang, Xuanjing",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2024",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-emnlp.628",
    doi = "10.18653/v1/2024.findings-emnlp.628",
    pages = "10704--10714",
    abstract = "In many PDF documents, the reading order of text blocks is missing, which can hinder machine understanding of the document{'}s content.Existing works try to extract one universal reading order for a PDF file.However, applications, like Retrieval Augmented Generation (RAG), require breaking long articles into sections and subsections for better indexing.For this reason, this paper introduces a new task and dataset, PDF-to-Tree, which organizes the text blocks of a PDF into a tree structure.Since a PDF may contain thousands of text blocks, far exceeding the number of words in a sentence, this paper proposes a transition-based parser that uses a greedy strategy to build the tree structure.Compared to parser for plain text, we also use multi-modal features to encode the parser state.Experiments show that our approach achieves an accuracy of 93.93{\%}, surpassing the performance of baseline methods by an improvement of 6.72{\%}.",
}
```
