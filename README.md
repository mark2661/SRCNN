# SRCNN
Implementation of the image super resolution convolutional neural network proposed in [Image Super-Resolution Using Deep Convolutional Networks](https://arxiv.org/abs/1501.00092 "Image Super-Resolution Using Deep Convolutional Networks").
## Training
```python
python Main.py --training-data #path to training set .h5 file
               --validation-data #path ro validation set .h5 file
               --lr #model learning rate (10e-4 default)
               --batch-size #dataloader batch size (16 default)
               --epochs #the desired amount of training epochs (100 default)
               --output-dir #path to the output directory
               --model-num #unique id number the the model (1 default) (optional)
```

## Testing
### Test sets
* [Set5](http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html)
* [Set14](http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html)
* [BSDS](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/)
* [Urban100](https://paperswithcode.com/dataset/urban100)
```python
python Test.py --test-set-path #comma seperated list of file paths to the desired test sets
               --model-weights-path #path to the trained model weights

## Results

## Prediction
```python predict.py --image-path #path to the low res image
                     --model-weights-path #path to the trained model weights
```
