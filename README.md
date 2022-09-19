# My-First-Neural-Network
This is, as the title suggests, my first dive into deep learning! It is a feed-forward neural network that I trained on the MNIST handwritten digits dataset as well as Google's QuickDraw! dataset. 
<br>**This is in no way a finished project, and it is fragile, but it's my own :)**

### To Use
(1) Download python3
(2) Install the required dependencies with
```pip install -r requirements.txt```
(3) Open Guesser.py and run it to open a pygame window to mess around with the pretrained network

### To train
This model requires 2 arrays to train: 
1) A numpy array of training data in the shape (Number of Input Nodes (784 for a grayscale 28x28 image)) X (Training Data Size) 
2) A numpy array of labels for your data indexed from 0-n. Eg for eleven doodles, the labels would be 0-10. The network expects the shape 1 X Number Training Data Size
to train the network: 
(1) Import NeuralNetwork into your project
(2) Initialize the network with 
```
myNetwork  = NeuralNetwork.NeuralNetwork(Train Data(NP ndArray), Train Labels(NP ndArray), [num first hidden layer nodes,...,num Nth Hidden layer nodes, num Output Nodes](integer[]),learning rate(float), training:validation ratio(range (0,1) float) default=0.8)
 ```
 (3) Train the network with ```myNetwork.train(batch size, number of iterations)```. This will also create acuracyLog.csv containing the validaton and testing acuracy per iteration<br>
 (4) Save the configuration data with ```myNetwork.saveConfiguration(output filename)```. This saves the configuration data to the output directory under the specified nameWn.csv, nameBn.csv <br>
 (5) Get predictions for 1 or more datapoints with ```myNetwork.classify(data)```. Like the training data, data is in the shape (N input nodes) X (N datapoints)<br>
 
 ### To Load a Configuration
 (1) If you dont have training data or dont want to give the network training data, there's no easy way to do that yet, sorry. Instead. Use this when initializing:
 ```myNetwork = NeuralNetwork.NeuralNetwork(np.zeros((Input Layer Size,1)),np.int32(np.zeros((1,1))),[Layer1 Size,...,LayerN Size,Output Size],LearnRate)```
 <br>
 (2) Load data using the command ```myNetwork.load("Path/fileName",Num of Layers to look for)``` Dont include the Wn.csv or theBn.
 (3) Done! Get predictions the same way as above. 
 
 



