# 1. Why negative sampling is required?

Earlier, we have implemented a softmax function to return the probability of the word and compare it to the real target. However,  when we use this approach this basically, every word is non-target over 99% of the time. So with the softmax, we end up saying almost every word except the targets is considered wrong.  Therefore, we simply take a sample of these wrong words.  

![image](https://user-images.githubusercontent.com/53164959/76927879-63f51780-6923-11ea-8b2c-f5cd8690e1f5.png)

# 2. Multiclass and Binary Classification


Recall, v on the right is the row vector of the weighting Matrix

