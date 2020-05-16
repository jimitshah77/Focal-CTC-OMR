# Focal-CTC-OMR  

## Focal CTC Loss Function
```python
def focal_ctc(alpha=0.5,gamma=2.0,targets,logits,seq_len):
      
    #FOCAL LOSS
    #This function computes Focal Loss
    #Inputs: alpha, gamma, targets, logits, seq_len
    #Default Values: alpha=0.5 and gamma=2.0
    #Output: loss
       
    ctc_loss = tf.nn.ctc_loss(labels=targets, inputs=logits, sequence_length=seq_len, time_major=True)
    p= tf.exp(-ctc_loss)
    focal_ctc_loss= tf.multiply(tf.multiply(alpha,tf.pow((1-p),gamma)),ctc_loss) #((alpha)*((1-p)**gamma)*(ctc_loss))
    loss = tf.reduce_mean(focal_ctc_loss)
      
return loss    
```
## Problem Statement  
Given a music sheet, usually in the form of an image, the goal of an OMR system is to use various vision algorithms to interpret the corresponding music symbols and later convert it into digital playable format of music. To solve this problem in an End-to-End manner, Convolutional Recurrent Neural Network (CRNN) architecture is used. It considers both spatial and sequential nature of this problem. CTC loss function is proved to be a favorable choice in these types of sequence problems as it trains the models directly from input images to their corresponding musical transcripts without the need for a frame-by-frame alignment between the image and the ground-truth thereby solving the purpose of End to-End training. Though traditional CTC seems to solve a major chunk of the problem, it suffers from some limitations due toUnbalanced Dataset. 
### Solution: Focal CTC 
## Data Visualization  
## Expriments and Results  

## Sampling the Dataset  
Script: sampling_1250.ipynb  

## Visualizing Sampled Dataset  
Script: visualization_of_sampled_dataset_1250.ipynb  
![python](/images/wordcloud2.png)  
![python](/images/frequencyVsNotes.png)  
![python](/images/boxPlot.png)  
![python](/images/barGraph.png)  
![python](/images/waffleChart.png)  

## Spliting Sampled_1250 Dataset in Training and Testing Dataset  
Script: train_test_split.ipynb  
![python](/images/testing_training.png)  


## Scripts
1. sampling_1250.ipynb :Used for Sampling the Dataset
2. visualization_of_sampled_dataset_1250.ipynb :
3. train_test_split.ipynb

## 
Base Code is taken from https://github.com/OMR-Research/tf-end-to-end
