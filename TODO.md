For now, I have trained the network using the Range-Doppler representation. However, the model started to overfit at some point, so I introduced gradient clipping to stabilize the training.

Other possible improvements could be:

- Freezing the weights of the detection head, or using a lower learning rate for the head compared to the backbone

- Using AdamW with weight decay for better regularization

- Increasing the dropout rate (e.g., up to 0.3)

- Applying data augmentation

