# Mortgage_Perceptron
Acyclic graph based approach to single layer perceptron to qualify people for a mortgage based on their info. 2 sets of training with 351724 training samples out of 439655, and test 87931 samples : accuracy 92%. With a max accuracy of 96%. 

Youtube link: https://youtu.be/7hdMRqZ86jc

II Real-world perceptron architecture: I based my perceptron off the light perceptron but for mortgage applications. It determines if someone is eligible for a mortgage based on income, race, gender, loan type, loan amount, property type, and loan purpose.
I found a dataset from 2015 containing NY mortgage application data and modified the light perceptron to work with it. It uses normalization to try to keep the data as relative as possible. Once the weights are randomly generated, the perceptron is trained to determine if the person qualifies or not. 1 to 2 are yes, 3 to 6 are no (action taken). If the guess is incorrect, I use the delta formula to calculate the change and multiply the change for each weight by the input value like in the powerpoint. 
 
•	T is expected outcome
•	o is observed output
•	x is val (or input value)
•	n is learning rate (I used a high learning rate of 0.5)
Then I added the delta to the old weight which results in the new weight. The key thing was sifting expected outputs of 1-6 to either 1 or -1 and normalizing the inputs for better distribution.
 
Z distribution would have probably been better. Either way the results came back, and they were amazing, very accurate on the test samples.
The data set consisted of 439655 samples of mortgage applications, so I divided it by 80% for training, and 20% for testing. The output of classification is below this code followed by the test code. The test code uses confusion matrix to calculate the accuracy.
 
 
Finally, when building the perceptron, I used a directed acyclic graph and each vertex had an object called neuron. Each perceptron had a number of input neurons connected to a soma. The soma is connected to the axon (output). There is also a hidden output which is the result of processing for both the light and the mortgage perceptron. I followed the powerpoint to determine how to compare our inputs in relation to weights to a threshold. This gives us either a positive or negative result.
If the sum of weights * inputs is greater than 0, it is positive, if it is less than 0, it is negative.
 
 
The following line of code makes it a DAG, instead of creating two edges, I only create one with a starting and end point.
        # self.vert_dict[frm].add_neighbor(self.vert_dict[to], cost)
        # self.vert_in_dict[frm].add_input(self.vert_in_dict[to],cost)
        self.vert_dict[to].add_neighbor(self.vert_dict[frm], cost)
Lastly, I add a bias neuron. That is how I did it. The DAG structure is the same as I used for both perceptrons, I just expanded it for the mortgage by adding more input neurons.


