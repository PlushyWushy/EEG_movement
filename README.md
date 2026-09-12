EEG movement project

Explain your data processing pipeline. 

I initially followed the CNN-GRU process quite closely. 
That process cleans the raw recordings (standard montage, notch filter, ICA for eye/muscle artifacts), narrows each one down to a handful of sensorimotor channel pairs, band-passes to the motor-relevant frequency range, and cuts the signal into per-trial windows, normalizing each subject against their own signal so amplitude differences between people don't leak in. SMOTE then rebalances the training set toward the rarer classes. However, I must've done something wrong with this preprocessing, as it seems that the networks I used with it  performed worse. So, I moved away from it in favor of feeding the networks more raw signals, as each of the transformer tests were run without preprocessing with all 64 channels, though SMOTE and other augmentations were still used at training time.

What your model can and cannot do. 

The situation the main reported accuracies describe are for imagined LH vs RH, meaning, if deployed in its current state, it would be able to differentiate whether the user had imagined squeezing their left hand vs their right hand. One condition for this model is the four second interval. Since it was trained using the specific 4 second intervals provided by this dataset, it may have more trouble with shorter imagined squeeze times.

That the result is not an artifact of how you measured it. 

I tailored the main experiment (imagined RH vs LH) to be directly comparable to Kumar, Tang, Yoo & Michmizos 2022, as they were the highest accuracy for that task I could find. For these tests, the train-val and test sets were split. For training and validation, 90 subjects were chosen. Thus, if any overfitting happened on these 90 subjects, the test accuracy would show. This is also how Kumar et al. performed their tests. 

For the full dataset task, I used the default individual splitting method, where each individual subject's data is split into train-val-test. Here, I originally allowed the model to guess "baseline" along with the other four classes, just like CNN-GRU did. However, this inflated the accuracy and was uncomparable to the papers I found in litreview, so I excluded that choice for the reported "full task" runs.  

What the appropriate baselines are + performance attribution.

Because of the way the data was taken, a given 4 second interval is affected by the previous 4 second intervals due to participant memory. So, with the attention mechanism, what I'm really decoding is what the current 4 second interval is given all the previous 4 second intervals. That is the core insight of this project. 

For other baselines, I used what I found in my literature review, along with my CNN-GRU and EEGnet implementations. 



How much of this is about the person rather than the task. These recordings come from many individuals, and individuals differ from each other in ways that have nothing to do with what they were imagining. Quantify how much this matters for your result.

When testing under the 90:13 split regime, the val accuracy during training did not deviate too much from the final test accuracy. This is evidence that, at least for those three seeds, the model generalized well on the chosen test participants based on the 90 train participants, meaning the individual may not matter too much for this model when it comes to imagined RH vs LH. 

Whether your model has learned or memorized. Where does it overfit, to what, and how do you know?

Since I used SMOTE and randaugment, it only tended to overfit slightly at the end. Specifically, the train accuracy tends to begin leaving the val accuracy behind at about 89% accuracy. Before that, however, there usually isn't any overfitting. This could be attributed to the augmentation mechanism running out of ways to help on specific trials and its exhaustion. Furthermore, the val set is never augmented, so perhaps at some point, the model starts getting extra points on the augmented trials that the val set doesn't give. 




Your evaluation and what it establishes. 

The final results were the evaluations of CNN-GRU, EEGnet, and the singleview transformer to SNN pipeline. I also provide an ablation for the attention mechanism that attends each four second interval to each one before it, showing that this mechanism yields a meaningful boost in accuracy. 

This ablation was conducted on the ANN singleview transformer, before a conversion to an SNN. A future ablation may compare attentionless vs with attention spiking architectures, but due to the training nature of this project (the attention mechanism gets an extra 200 epochs of training, making a fair comparison difficult), only preconversion metrics were used for the ablation. However, since accuracy of postconversion is meant to match the accuracy of the preconversion, comparing preconversion results is a good measure for postconversion accuracies as well. 

The accuracies of each test can be found in accuracy_tracker.txt. As that .txt file shows, the spikformer achieved an average 3-seed accuracy of 0.8921. This, to my knowledge, is the state-of-the-art performance of a spiking neural network on this dataset. I found a few other papers utilizing SNNs for this dataset, and their accuracies are compared below:

(Kumar, Tang, Yoo & Michmizos 2022): 80.65 ± 3.83 % for the left-right imagined. They did not run it on the full dataset. 

(Garg, Song, Plessnig, Savoia & Bégon-Lours 2026) 80.39 ± 2.98 % for left-right imagined. They also omitted a run on the full dataset. 

(Raja Sekhar Banovoth, Kadambari K V) 73.65% for four class.

(Yulin Li, Liangwei Fan) 67.24% for four class

The four class tests are currently running, though I expect it to beat at least Li and Fan's 67.24%. This evaluation establishes a novel method of using spiking networks to analyze these signals, paving way for ultra low power onboard EEG analysis. 


One design decision. 

The most important decision made during this project was based on the insight that past 4-second intervals may affect future 4-second intervals, inspiring the interval level attention mechanism. As shown by the ablation, this alone boosted the accuracy by 3.94%. 



Your weakest point. 

The purpose of creating an SNN to analyze EEG is to take advantage of its high efficiency on neuromorphic hardware. However, I unfortunately do not have access to any actual neuromorphic chips. The weakest point of this project, I believe, is the lack of testing on true neuromorphic hardware. Furthermore, I believe not many chips currently are able to support the spiking attention, so compatibility may also be a large issue. 


Something we did not ask about.

Of course, as the ablation showed, the key insight of this project was the intervals affecting each other down the line. Beyond this, I found that preprocessing this data tended not to help as much as I expected. I originally followed the CNN-GRU's preprocessing pipeline, with their band filtering and ICA. However, I tried it without, and the networks I tested performed better. Perhaps the preprocessing was implemented incorrectly. Nonetheless, I proceeded without. 



What you would do next with more time and more compute.

1. 5 seeded runs instead of 3. Spiking neural network performance can be very random.

2. Tests on a neuromorphic chip. This would confirm the energy savings from switching from ANN to SNN. 

3. More EEG datasets. One dataset is definitely not enough. 

4. Further optimization of the Spikformer. The architecture of the transformer and subsequently the Spikformer were put together on a time crunch due to the deadline of this project; if I had more time, I'd go through a more methodical search of architectures. 

5. Multiseeded and better baselines. This goes without saying, but to establish this method as a real SOTA, I'd need much stronger baselines and better comparisons to more papers, which would come as I add more datasets.

6. Local learning for deployment? This one is still just a hypothesis, but perhaps, when actually deployed on a neuromorphic chip, the network can engage in a sort of life-long learning to personalize itself to the user. Backprop isn't very good for neuromorphic chips, so we'd use local learning methods like STDP. 


AI use.  

Claude code was used to assist in coding this project. My decision to move to a transformer was fueled both by the cross-interval interference and my past experience in EEG. In the litreview folder, there is a paper named eeg2text. This paper was one of my first exposures to EEG. I spent a few months my junior year trying to implement a multiview transformer to analyze text. Although it didn't work out as well as I'd hoped, I took my experience with EEG to text decoding to this project. 




Trained on apple silicon M5 Max





Notes throughout project:
Baselines:
Simple MLP
EEGnet
CNN
CNN-gru
Minirocket
Tests:
Multiview transformer with masked token pretraining
Spiking variant

It seems that CNN-gru without the bulky preprocessing pipeline outperforms itself when the preprocessing is turned on. 

Though, on the full dataset, the multiview transformer still outperforms. 

Multiview transformer with masked token pretraining notes:

Data should be fed in sequentially; each prediction should be made on the input of multiple four second intervals, as humans have memory; subjects remembering what they did earlier in the recording affects the signal of the current action. This way, we also take advantage of the transformer's strength in context. 





Spiking implementation:

CNN-gru but spiking; good if low power/onboard-with-eeg classification is ever needed. perhaps it can pretrain (like masked token prediction) constantly on the user's EEG signals using STDP/other local rules to more personalize the classification.

Turns out conversion from both CNN-gru and EEGnet don't work well

conversion from multiview transformer? CNN encoders can convert but attention can't