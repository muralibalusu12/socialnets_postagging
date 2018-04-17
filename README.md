# socialnets_postagging
Code for the paper [Stylistic Variation in Part of Speech Tagging](https://www.cc.gatech.edu/~jeisenst/papers/social-attention-naacl-style-2018.pdf)

## Dependencies
- python 3.4 and above (2.7 should also work fine)
- dynet 1.0 or above
- scipy
- matplotlib
- [emoji](https://pypi.org/project/emoji/)

## To run the baseline Bi-LSTM Tagger
```
python ensemble_train.py --no-ensemble
```

## To run just a plain ensemble of Bi-LSTM Tagger
```
python ensemble_train.py --just-ensemble
```

## To run the Social Attention Tagger using the follow network with 4 basis models.
```
python ensemble_train.py --network=follow --num-basis=4
```

Please feel free to [contact me](muraliraghubabu1994@gmail.com) if you have any questions on the code or the paper.

If you use this code, please cite our paper:
```
@inproceedings{balusu2018social,
  title = {Stylistic Variation in Social Media Part-of-Speech Tagging},
  author = {Murali Raghu Babu Balusu and Taha Merghani and Jacob Eisenstein},
  booktitle = {Proceedings of {NAACL} workshop on stylistic variation},
  year = {2018},
  url = {../papers/social-attention-naacl-style-2018.pdf}
}
```




