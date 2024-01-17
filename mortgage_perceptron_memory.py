class Vertex:
    def __init__(self, node):
        self.id = node
        self.adjacent = {}
        self.inbound={}

    def __str__(self):
        return str(self.id) + ' adjacent: ' + str([x.id for x in self.adjacent])

    def __lt__(self,other):
        return self.id<other.id

    def add_neighbor(self, neighbor, weight=0):
        self.adjacent[neighbor] = weight

    def add_input(self,neighbor,weight=0):
        self.inbound[neighbor]=weight

    def get_connections(self):
        connections= list(self.adjacent.keys())
        weights=[]
        for i in connections:
            weights.append(self.get_weight(i))
        return connections,weights
    
    def get_inbound_edges(self):
        return self.inbound.keys()

    def get_id(self):
        return self.id

    def get_weight(self, neighbor):
        return self.adjacent[neighbor]
    
    def set_weight(self,neighbor, new_weight):
        self.adjacent[neighbor]=new_weight

class Graph:
    def __init__(self):
        self.vert_dict = {}
        self.num_vertices = 0
        self.vert_in_dict={}
        self.num_in_vertices=0

    def __iter__(self):
        return iter(self.vert_dict.values())

    def add_vertex(self, node):
        self.num_vertices = self.num_vertices + 1
        new_vertex = Vertex(node)
        self.vert_dict[node] = new_vertex
        return new_vertex
    
    def add_in_vertex(self, node):
        self.num_in_vertices = self.num_in_vertices + 1
        new_vertex = Vertex(node)
        self.vert_in_dict[node] = new_vertex
        return new_vertex

    def get_vertex(self, n):
        if n in self.vert_dict:
            return self.vert_dict[n]
        else:
            return None

    def add_edge(self, frm, to, cost = 0):
        if frm not in self.vert_dict:
            self.add_vertex(frm)
            self.add_in_vertex(frm)
        if to not in self.vert_dict:
            self.add_vertex(to)

        # self.vert_dict[frm].add_neighbor(self.vert_dict[to], cost)
        # self.vert_in_dict[frm].add_input(self.vert_in_dict[to],cost)
        self.vert_dict[to].add_neighbor(self.vert_dict[frm], cost)

    def get_vertices(self):
        return self.vert_dict.keys()

    def process(self,node_list,weight_list,threshold):
        total=0
        total_nodes=1/(len(node_list))
        for count in range(len(node_list)):
            id=node_list[count].get_id()#get neuron
            val=id.get_value()#get neuron value
            total+=(weight_list[count]*val)
        if total>threshold:return total,1
        else: return total,-1
#neuron
class Neuron:
    def __init__(self,value):
        self.value=value

    def get_value(self):
        return self.value
    def set_value(self,value):
        self.value=value

#perceptron declaration
def populate_perceptron(input_neurons,bias,input_weights):
    #verteces
    for v in input_neurons:
        #neuron space
        perceptron.add_vertex(v)

    perceptron.add_vertex(soma)
    perceptron.add_vertex(axon)
    perceptron.add_edge(bias_neuron,soma,bias)
    count=1
    for v in input_neurons:
        perceptron.add_edge(v,soma,input_weights[count])
        count+=1
        #output edge
    perceptron.add_edge(soma,axon,0)
    

import random
#perceptron declaration
def populate_perceptron_random(input_neurons,bias):
    #verteces
    for v in input_neurons:
        #neuron space
        perceptron.add_vertex(v)

    perceptron.add_vertex(soma)
    perceptron.add_vertex(axon)
    perceptron.add_edge(bias_neuron,soma,bias)
    count=1
    for v in input_neurons:
        rand_int=random.randint(0,10)
        rand=rand_int*0.1
        perceptron.add_edge(v,soma,rand)
        count+=1
        #output edge
    perceptron.add_edge(soma,axon,0)

#epoch
def epoch(threshold):
    print('Begin epoch')
    # print(f'Weights in epoch: {perceptron.get_vertex(soma).get_weight(vertices[0])}')
    output_layer=perceptron.get_vertex(soma)
    vertices,weights=output_layer.get_connections()
    hidden_output, observed_output=perceptron.process(vertices,weights,threshold)
    print(f'current weights {weights}')
    print(f'hidden output {hidden_output}')
    axon.value=observed_output
    print(f'axon {axon.get_value()}')
    return vertices, observed_output

def min_max_normalization(data):
    min_val = min(data)
    max_value = max(data)
    normalized_data = [(x - min_val) / (max_value - min_val) for x in data]
    return normalized_data

#learning
def learn(learning_rate, expected_outcome, observed_output):
    print(f'Begin learn')
    old_weights=[]
    delta=[]
    new_weights=[]
    change=learning_rate*(expected_outcome-observed_output)
    values=[]
    
    for i in range(len(vertices)):
        old_weights.append(perceptron.get_vertex(soma).get_weight(vertices[i]))
        val=vertices[i].get_id().get_value()
        values.append(val)
        print(val)
        i+=1
    values=min_max_normalization(values)#normalization
    print(f'old weights {old_weights}')
    print(f'Delta {delta}')
    for n in range(len(vertices)):
        new_weight=round((old_weights[n]+(values[n]*change)),2)
        new_weights.append(new_weight)
        perceptron.get_vertex(soma).set_weight(vertices[n],new_weights[n])
    print(f'new weights {new_weights}')
    
    return ",".join([str(i) for i in new_weights])

import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv('train.csv')

# Select the relevant columns
selected_columns = [
    'applicant_income_000s',#0
    'applicant_race_1',#1
    'applicant_sex', #2
    'loan_purpose',#3
    'loan_type', #4
    'loan_amount_000s',#5
    'property_type',#6
    'action_taken',#output 7
    'respondent_id',#8
    'sequence_number',#9
    'applicant_race_name_1',#10
    'applicant_sex_name',#11
    'loan_purpose_name',#12
    'loan_type_name',#13
]

# Create a new DataFrame with only the selected columns
selected_df = df[selected_columns]

# Drop rows with 'nan' values in any of the selected columns
selected_df = selected_df.dropna()

# Convert 'sequence_number' column to string type
selected_df['sequence_number'] = selected_df['sequence_number'].astype(str)

# Display the resulting DataFrame
print(selected_df)
sample = selected_df.values.tolist()
x=len(sample)
y=len(sample[0])
#initialize DAG
perceptron=Graph()
soma=Neuron(None)
axon=Neuron(None)
bias_neuron=Neuron(1)
input_neurons=[]
#read input weights
f=open('application_weights.txt','r')
line= f.read()
text_read = line.split(',')
input_weights=list(map(float,text_read))

bias=input_weights[0]
#neuron_space
for i in range(len(sample[0])):
    if type(sample[0][i])is not str:
        input_neurons.append(Neuron(0))

# populate_perceptron_random(input_neurons,bias)
populate_perceptron(input_neurons,bias,input_weights)
# Convert dictionary items to a list of tuples
keys_list = list(perceptron.vert_dict.keys())

for n in range(len(sample)):
    #supplemenary variables not in perceptron
    j=sample[n][8]#res id
    k=sample[n][9]#seq number
    l=sample[n][10]#race name
    m=sample[n][11]#gender name
    p=sample[n][12]#purpose name
    q=sample[n][13]#loan name

    w=0
    length=len(keys_list)
    while w < length-1:
        key=keys_list[w]
        #neuron_space
        perceptron.vert_dict[key].get_id().set_value(sample[n][w])
        # print(perceptron.vert_dict[key].get_id().get_value())
        w+=1

    print(f'Income: {sample[n][0]} Loan Amount: {sample[n][5]} {j} {k} {l} {m} {p} {q}')
    expected_output=sample[n][7]#output neuron
    expected_outcome=0
    if expected_output in range(1,3):
        expected_outcome=1
    else:
        expected_outcome=-1
    #epoch declarations:
    threshold=0
    #learning declarations:
    learning_rate=0.5
    observed_output=expected_output-1
    vertices,observed_output=epoch(threshold)
    if observed_output!=expected_outcome:
        print('incorrect')
        post_weights=learn(learning_rate, expected_outcome, observed_output)
    else:
        print('correct\n')    

f=open('application_weights.txt','w')
f.truncate(0)
f.write(post_weights)
f.close()
