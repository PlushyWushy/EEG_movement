EEG movement project

Your evaluation and what it establishes. 
Baselines:
Simple MLP
EEGnet
CNN
CNN-gru
Minirocket
Tests:
Multiview transformer with masked token pretraining
Spiking variant

It seems that CNN-gru without the bulky preprocessing pipeline outperforms itself when the preprocessing is turned on. Perhaps preprocessing takes away too much information, preventing the richer representation preprocessless training yields.

Though, on the full dataset, the multiview transformer still outperforms. 

Multiview transformer with masked token pretraining notes:

Data should be fed in sequentially; each prediction should be made on the input of multiple four second intervals, as humans have memory; subjects remembering what they did earlier in the recording affects the signal of the current action. This way, we also take advantage of the transformer's strength in context. 


Todo:
Pretraining ablation (save hyperparam (percent, epochs, etc.) search for future work)


Spiking implementation:

CNN-gru but spiking; good if low power/onboard-with-eeg classification is ever needed. perhaps it can pretrain (like masked token prediction) constantly on the user's EEG signals using STDP/other local rules to more personalize the classification.

Turns out conversion from both CNN-gru and EEGnet don't work well

conversion from multiview transformer? CNN encoders can convert but attention can't


One design decision. 

Your weakest point. 

Something we did not ask about.

What you would do next with more time and more compute.

AI use. why 




Trained on apple silicon M5 Max